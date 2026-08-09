from __future__ import annotations

# ruff: noqa: ANN201, BLE001, PLR0913
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
from torch import Tensor
from torch._inductor import config as inductor_config

from luxai2021.env.agent import Agent
from luxai2021.imitation.agent import BehaviorCloningAgent, FirstPlaceAgent
from luxai2021.imitation.first_place import predict_first_place
from luxai2021.imitation.schema import snapshot_from_game
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent

if TYPE_CHECKING:
    from collections.abc import Callable

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
_COMPILE_CALIBRATION_LOCK = threading.Lock()


@dataclass
class _Pending(Generic[RequestT, ResponseT]):
    payload: RequestT
    ready: threading.Event
    enqueued_at: float
    result: ResponseT | None = None
    error: BaseException | None = None


class InferenceFuture(Generic[ResponseT]):
    def __init__(self, pending: _Pending[Any, ResponseT]) -> None:
        self.pending = pending

    def result(self) -> ResponseT:
        self.pending.ready.wait()
        if self.pending.error is not None:
            raise RuntimeError("Batched inference failed") from self.pending.error
        return self.pending.result  # type: ignore[return-value]


class _MappedInferenceFuture(Generic[RequestT, ResponseT]):
    def __init__(self, source: InferenceFuture[RequestT], transform: Callable[[RequestT], ResponseT]) -> None:
        self.source = source
        self.transform = transform

    def result(self) -> ResponseT:
        return self.transform(self.source.result())


class InferenceBatcher(Generic[RequestT, ResponseT]):
    """Collect blocking or future-backed requests from environment threads into GPU batches."""

    def __init__(
        self,
        batch_fn: Callable[[list[RequestT]], list[ResponseT]],
        *,
        max_batch_size: int,
        wait_seconds: float = 0.002,
        name: str,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.wait_seconds = max(0.0, wait_seconds)
        self.condition = threading.Condition()
        self.pending: list[_Pending[RequestT, ResponseT]] = []
        self.expected_participants: int | None = None
        self.closed = False
        self.batch_sizes: list[int] = []
        self.batch_targets: list[int] = []
        self.inference_seconds = 0.0
        self.queue_wait_seconds = 0.0
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit_async(self, payload: RequestT) -> InferenceFuture[ResponseT]:
        pending: _Pending[RequestT, ResponseT] = _Pending(payload, threading.Event(), time.monotonic())
        with self.condition:
            if self.closed:
                raise RuntimeError("Inference batcher is closed")
            self.pending.append(pending)
            self.condition.notify_all()
        return InferenceFuture(pending)

    def submit(self, payload: RequestT) -> ResponseT:
        return self.submit_async(payload).result()

    @staticmethod
    def _validate_results(results: list[ResponseT], expected: int) -> None:
        if len(results) != expected:
            raise RuntimeError("Batched inference returned the wrong number of results")

    def _run(self) -> None:  # noqa: C901
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if self.closed and not self.pending:
                    return
                deadline = self.pending[0].enqueued_at + self.wait_seconds
                while True:
                    target = min(self.max_batch_size, self.expected_participants or self.max_batch_size)
                    target = max(1, target)
                    if len(self.pending) >= target:
                        break
                    if self.expected_participants is not None:
                        self.condition.wait()
                        continue
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    self.condition.wait(timeout=timeout)
                pending = self.pending[: self.max_batch_size]
                del self.pending[: len(pending)]
                batch_target = target
            dequeued_at = time.monotonic()
            started_at = time.perf_counter()
            try:
                results = self.batch_fn([request.payload for request in pending])
                self._validate_results(results, len(pending))
                for request, result in zip(pending, results):
                    request.result = result
            except BaseException as error:
                for request in pending:
                    request.error = error
            finally:
                self.inference_seconds += time.perf_counter() - started_at
                self.batch_sizes.append(len(pending))
                self.batch_targets.append(batch_target)
                self.queue_wait_seconds += sum(dequeued_at - request.enqueued_at for request in pending)
                for request in pending:
                    request.ready.set()

    @contextmanager
    def batch_scope(self, participants: int):
        if participants < 1:
            raise ValueError("participants must be positive")
        with self.condition:
            if self.expected_participants is not None:
                raise RuntimeError("Inference batch scopes cannot overlap")
            self.expected_participants = participants
            self.condition.notify_all()
        try:
            yield self
        finally:
            with self.condition:
                self.expected_participants = None
                self.condition.notify_all()

    def participant_done(self) -> None:
        with self.condition:
            if self.expected_participants is None:
                return
            self.expected_participants = max(0, self.expected_participants - 1)
            self.condition.notify_all()

    def metrics(self) -> dict[str, float]:
        count = len(self.batch_sizes)
        return {
            "batches": float(count),
            "samples": float(sum(self.batch_sizes)),
            "mean_batch_size": float(sum(self.batch_sizes) / count) if count else 0.0,
            "max_batch_size": float(max(self.batch_sizes, default=0)),
            "mean_batch_fill_ratio": (
                float(sum(size / max(target, 1) for size, target in zip(self.batch_sizes, self.batch_targets)) / count)
                if count
                else 0.0
            ),
            "inference_seconds": self.inference_seconds,
            "queue_wait_seconds": self.queue_wait_seconds,
        }

    def reset_metrics(self) -> None:
        self.batch_sizes.clear()
        self.batch_targets.clear()
        self.inference_seconds = 0.0
        self.queue_wait_seconds = 0.0

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Inference batcher did not stop")


def _cpu_output_batch(output: dict[str, Tensor]) -> dict[str, Tensor]:
    """Copy each entity tensor once instead of once per sample and entity."""
    return {name: logits.float().cpu() for name, logits in output.items()}


def _split_cpu_outputs(output: dict[str, Tensor], batch_size: int) -> list[dict[str, Tensor]]:
    return [{name: logits[index : index + 1] for name, logits in output.items()} for index in range(batch_size)]


def resolve_rollout_precision(requested: str, device: torch.device) -> tuple[str, torch.dtype]:
    if requested not in {"auto", "fp32", "bf16", "fp16"}:
        message = f"Unsupported rollout precision: {requested}"
        raise ValueError(message)
    if device.type != "cuda":
        return "fp32", torch.float32
    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        return "fp32", torch.float32
    return requested, {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[requested]


def configure_rollout_determinism(device: torch.device) -> None:
    """Keep repeated inference stable without forcing unsupported deterministic ops."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class _ObservationStager:
    def __init__(self, device: torch.device, max_batch_size: int, *, pad_batches: bool) -> None:
        self.device = device
        self.max_batch_size = max_batch_size
        self.pad_batches = pad_batches
        self.buffer: Tensor | None = None

    def stage(self, observations: list[Tensor]) -> tuple[Tensor, int]:
        sample_count = len(observations)
        target_count = self.max_batch_size if self.pad_batches else sample_count
        if self.device.type != "cuda":
            batch = torch.stack(observations)
            if target_count > sample_count:
                padding = torch.zeros(
                    (target_count - sample_count, *batch.shape[1:]),
                    dtype=batch.dtype,
                )
                batch = torch.cat((batch, padding), dim=0)
            return batch.to(self.device), sample_count
        expected_shape = (self.max_batch_size, *observations[0].shape)
        if self.buffer is None or tuple(self.buffer.shape) != expected_shape:
            self.buffer = torch.empty(expected_shape, dtype=observations[0].dtype, pin_memory=True)
        for index, observation in enumerate(observations):
            self.buffer[index].copy_(observation)
        if target_count > sample_count:
            self.buffer[sample_count:target_count].zero_()
        batch = self.buffer[:target_count].to(self.device, non_blocking=True)
        return batch.contiguous(memory_format=torch.channels_last), sample_count


class _CompiledInference:
    def __init__(
        self,
        module: torch.nn.Module,
        device: torch.device,
        requested: str,
        *,
        auto_eligible: bool = True,
    ) -> None:
        if requested not in {"auto", "on", "off"}:
            message = f"Unsupported rollout compile mode: {requested}"
            raise ValueError(message)
        self.eager = module
        self.device = device
        self.requested = requested
        self.effective = "off"
        self.fallback_reason: str | None = None
        self.error_detail: str | None = None
        self.compiled: torch.nn.Module | None = None
        self.compile_attempts = 0
        self.compile_seconds = 0.0
        self.calibrated = (
            requested == "off" or device.type != "cuda" or (requested == "auto" and not auto_eligible)
        )
        if device.type != "cuda" and requested != "off":
            self.fallback_reason = "compile_requires_cuda"
        elif requested == "auto" and not auto_eligible:
            self.fallback_reason = "auto_compile_requires_static_batches"

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _make_compiled(self) -> torch.nn.Module:
        if self.compiled is None:
            # Handle callers that imported Inductor before luxai2021.rl. A pool
            # already configured for fork is unsafe once rollout threads exist.
            if inductor_config.worker_start_method == "fork":
                inductor_config.worker_start_method = "subprocess"
            self.compile_attempts += 1
            self.compiled = torch.compile(self.eager, mode="reduce-overhead")
        return self.compiled

    def __call__(self, batch: Tensor) -> object:
        if self.calibrated:
            model = self.compiled if self.effective == "on" else self.eager
            return model(batch)
        with _COMPILE_CALIBRATION_LOCK:
            if self.calibrated:
                model = self.compiled if self.effective == "on" else self.eager
                return model(batch)
            calibration_started = time.perf_counter()
            try:
                compiled = self._make_compiled()
                if self.requested == "on":
                    self.effective = "on"
                    self.calibrated = True
                    return compiled(batch)
                compiled(batch)
                self._synchronize(self.device)
                eager_started = time.perf_counter()
                for _ in range(3):
                    self.eager(batch)
                self._synchronize(self.device)
                eager_seconds = time.perf_counter() - eager_started
                compiled_started = time.perf_counter()
                for _ in range(3):
                    compiled(batch)
                self._synchronize(self.device)
                compiled_seconds = time.perf_counter() - compiled_started
                if compiled_seconds <= eager_seconds * 0.9:
                    self.effective = "on"
                else:
                    self.fallback_reason = "compiled_forward_improvement_below_10_percent"
                self.calibrated = True
                model = compiled if self.effective == "on" else self.eager
                return model(batch)
            except Exception as error:
                self.effective = "off"
                self.calibrated = True
                self.fallback_reason = f"compile_failed:{type(error).__name__}"
                self.error_detail = repr(error).replace("\n", " ")[:500]
                return self.eager(batch)
            finally:
                self.compile_seconds += time.perf_counter() - calibration_started


class ActorCriticBatcher:
    def __init__(
        self,
        actor_critic: FullTurnActorCritic,
        device: torch.device,
        max_batch_size: int,
        *,
        name: str,
        wait_seconds: float = 0.002,
        precision: str = "auto",
        compile_mode: str = "off",
        pad_batches: bool = False,
        tta: str = "rot180",
    ) -> None:
        if tta not in {"none", "rot180"}:
            message = f"Unsupported actor-critic TTA: {tta}"
            raise ValueError(message)
        self.actor_critic = actor_critic
        self.device = device
        self.tta = tta
        self.precision, self.autocast_dtype = resolve_rollout_precision(precision, device)
        self.stager = _ObservationStager(device, max_batch_size, pad_batches=pad_batches)
        effective_compile_mode = compile_mode if tta == "none" else "off"
        self.inference_model = _CompiledInference(
            actor_critic,
            device,
            effective_compile_mode,
            auto_eligible=pad_batches,
        )
        self.compile_requested = compile_mode
        self.compile_fallback_reason = "compile_requires_tta_none" if tta != "none" and compile_mode != "off" else None
        self.stage_seconds = {"host_stage_and_h2d_submit": 0.0, "forward": 0.0, "device_to_host": 0.0}

        def infer(observations: list[Tensor]) -> list[tuple[dict[str, Tensor], Tensor]]:
            started_at = time.perf_counter()
            batch, sample_count = self.stager.stage(observations)
            self.stage_seconds["host_stage_and_h2d_submit"] += time.perf_counter() - started_at
            forward_started = time.perf_counter()
            autocast_enabled = device.type == "cuda" and self.autocast_dtype != torch.float32
            with (
                torch.inference_mode(),
                torch.autocast(
                    device.type,
                    dtype=self.autocast_dtype,
                    enabled=autocast_enabled,
                ),
            ):
                output, values = (
                    self.inference_model(batch) if tta == "none" else actor_critic.forward_tta(batch)
                )
            self.stage_seconds["forward"] += time.perf_counter() - forward_started
            copy_started = time.perf_counter()
            cpu_output = _cpu_output_batch(output)
            cpu_values = values.float().cpu()
            self.stage_seconds["device_to_host"] += time.perf_counter() - copy_started
            split = _split_cpu_outputs(cpu_output, sample_count)
            return [(split[index], cpu_values[index]) for index in range(sample_count)]

        self.batcher = InferenceBatcher(
            infer,
            max_batch_size=max_batch_size,
            wait_seconds=wait_seconds,
            name=name,
        )

    def submit(self, observation: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        return self.batcher.submit(observation)

    def submit_async(self, observation: Tensor) -> InferenceFuture[tuple[dict[str, Tensor], Tensor]]:
        return self.batcher.submit_async(observation)

    def batch_scope(self, participants: int):
        return self.batcher.batch_scope(participants)

    def participant_done(self) -> None:
        self.batcher.participant_done()

    def metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = self.batcher.metrics()
        metrics.update(
            {
                "precision": self.precision,
                "compile_requested": self.compile_requested,
                "compile_effective": self.inference_model.effective,
                "compile_fallback_reason": self.compile_fallback_reason or self.inference_model.fallback_reason,
                "compile_error_detail": self.inference_model.error_detail,
                "compile_attempts": self.inference_model.compile_attempts,
                "compile_seconds": self.inference_model.compile_seconds,
                "stage_seconds": dict(self.stage_seconds),
                "peak_cuda_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else None
                ),
            }
        )
        return metrics

    def reset_metrics(self) -> None:
        self.batcher.reset_metrics()
        self.stage_seconds = dict.fromkeys(self.stage_seconds, 0.0)

    def close(self) -> None:
        self.batcher.close()


class BehaviorCloningBatcher:
    def __init__(
        self,
        prototype: BehaviorCloningAgent,
        max_batch_size: int,
        *,
        name: str,
        wait_seconds: float = 0.002,
        precision: str = "auto",
        compile_mode: str = "off",
        pad_batches: bool = False,
    ) -> None:
        self.prototype = prototype
        self.precision, self.autocast_dtype = resolve_rollout_precision(precision, prototype.device)
        self.stager = _ObservationStager(prototype.device, max_batch_size, pad_batches=pad_batches)
        compiled_mode = compile_mode if prototype.tta == "none" else "off"
        self.inference_model = _CompiledInference(
            prototype.model,
            prototype.device,
            compiled_mode,
            auto_eligible=pad_batches,
        )
        self.compile_requested = compile_mode
        self.compile_fallback_reason = (
            "compile_requires_tta_none" if prototype.tta != "none" and compile_mode != "off" else None
        )
        self.stage_seconds = {"host_stage_and_h2d_submit": 0.0, "forward": 0.0, "device_to_host": 0.0}

        def infer(observations: list[Tensor]) -> list[dict[str, Tensor]]:
            started_at = time.perf_counter()
            batch, sample_count = self.stager.stage(observations)
            self.stage_seconds["host_stage_and_h2d_submit"] += time.perf_counter() - started_at
            forward_started = time.perf_counter()
            autocast_enabled = prototype.device.type == "cuda" and self.autocast_dtype != torch.float32
            with (
                torch.inference_mode(),
                torch.autocast(
                    prototype.device.type,
                    dtype=self.autocast_dtype,
                    enabled=autocast_enabled,
                ),
            ):
                output = (
                    self.inference_model(batch) if prototype.tta == "none" else prototype._predict(batch)  # noqa: SLF001
                )
            self.stage_seconds["forward"] += time.perf_counter() - forward_started
            copy_started = time.perf_counter()
            output = _cpu_output_batch(output)
            self.stage_seconds["device_to_host"] += time.perf_counter() - copy_started
            return _split_cpu_outputs(output, sample_count)

        self.batcher = InferenceBatcher(
            infer,
            max_batch_size=max_batch_size,
            wait_seconds=wait_seconds,
            name=name,
        )

    def make_agent(self) -> BehaviorCloningAgent:
        prototype = self.prototype
        submit = self.batcher.submit

        class BrokeredBehaviorCloningAgent(BehaviorCloningAgent):
            def __init__(self) -> None:
                Agent.__init__(self)
                self.device = torch.device("cpu")
                self.model = prototype.model
                self.checkpoint = prototype.checkpoint
                self.tta = prototype.tta
                self.rollout_batcher = self_outer

            def _predict(self, observation: Tensor) -> dict[str, Tensor]:
                return submit(observation[0].cpu())

        self_outer = self
        return BrokeredBehaviorCloningAgent()

    def batch_scope(self, participants: int):
        return self.batcher.batch_scope(participants)

    def participant_done(self) -> None:
        self.batcher.participant_done()

    def metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = self.batcher.metrics()
        metrics.update(
            {
                "precision": self.precision,
                "compile_requested": self.compile_requested,
                "compile_effective": self.inference_model.effective,
                "compile_fallback_reason": self.compile_fallback_reason or self.inference_model.fallback_reason,
                "compile_error_detail": self.inference_model.error_detail,
                "compile_attempts": self.inference_model.compile_attempts,
                "compile_seconds": self.inference_model.compile_seconds,
                "stage_seconds": dict(self.stage_seconds),
            }
        )
        return metrics

    def reset_metrics(self) -> None:
        self.batcher.reset_metrics()
        self.stage_seconds = dict.fromkeys(self.stage_seconds, 0.0)

    def close(self) -> None:
        self.batcher.close()


class FirstPlaceBatcher:
    def __init__(
        self,
        prototype: FirstPlaceAgent,
        max_batch_size: int,
        *,
        name: str,
        wait_seconds: float = 0.002,
        precision: str = "auto",
        compile_mode: str = "off",
        pad_batches: bool = False,
    ) -> None:
        self.prototype = prototype
        self.max_batch_size = max_batch_size
        self.pad_batches = pad_batches
        self.precision, self.autocast_dtype = resolve_rollout_precision(precision, prototype.device)
        self.compile_requested = compile_mode
        self.stage_seconds = {"encode_forward_and_device_to_host": 0.0}

        def infer(snapshots: list[Any]) -> list[dict[str, Tensor]]:
            sample_count = len(snapshots)
            if self.pad_batches and sample_count < self.max_batch_size:
                snapshots = [*snapshots, *([snapshots[-1]] * (self.max_batch_size - sample_count))]
            started_at = time.perf_counter()
            output = predict_first_place(
                prototype.model,
                snapshots,
                device=prototype.device,
                rot180=prototype.tta == "rot180",
                amp_dtype=self.autocast_dtype,
            )
            self.stage_seconds["encode_forward_and_device_to_host"] += time.perf_counter() - started_at
            return _split_cpu_outputs(output, sample_count)

        self.batcher = InferenceBatcher(
            infer,
            max_batch_size=max_batch_size,
            wait_seconds=wait_seconds,
            name=name,
        )

    def make_agent(self) -> FirstPlaceAgent:
        prototype = self.prototype
        submit = self.batcher.submit

        class BrokeredFirstPlaceAgent(FirstPlaceAgent):
            def __init__(self) -> None:
                Agent.__init__(self)
                self.device = torch.device("cpu")
                self.model = prototype.model
                self.checkpoint = prototype.checkpoint
                self.tta = prototype.tta
                self.rollout_batcher = self_outer

            def process_turn(self, game: object, team: int) -> list[object]:
                snapshot = snapshot_from_game(game)
                output = submit(snapshot)
                selected_output = self._select_team_output(output, team)
                unit_choices = self._choose_units(game, team, snapshot, selected_output, 0, 0)
                actions = self._choices_to_actions(unit_choices, team)
                actions.extend(self._city_actions(game, team, snapshot, selected_output, 0, 0))
                return actions

        self_outer = self
        return BrokeredFirstPlaceAgent()

    def submit_team(self, snapshot: object, team: int) -> dict[str, Tensor]:
        output = self.batcher.submit(snapshot)
        return FirstPlaceAgent._select_team_output(output, team)  # noqa: SLF001

    def submit_team_async(
        self,
        snapshot: object,
        team: int,
    ) -> _MappedInferenceFuture[dict[str, Tensor], dict[str, Tensor]]:
        return _MappedInferenceFuture(
            self.batcher.submit_async(snapshot),
            lambda output: FirstPlaceAgent._select_team_output(output, team),  # noqa: SLF001
        )

    def batch_scope(self, participants: int):
        return self.batcher.batch_scope(participants)

    def participant_done(self) -> None:
        self.batcher.participant_done()

    def metrics(self) -> dict[str, object]:
        metrics: dict[str, object] = self.batcher.metrics()
        metrics.update(
            {
                "precision": self.precision,
                "compile_requested": self.compile_requested,
                "compile_effective": "off",
                "compile_fallback_reason": (
                    "first_place_compile_unsupported" if self.compile_requested != "off" else None
                ),
                "stage_seconds": dict(self.stage_seconds),
            }
        )
        return metrics

    def reset_metrics(self) -> None:
        self.batcher.reset_metrics()
        self.stage_seconds = dict.fromkeys(self.stage_seconds, 0.0)

    def close(self) -> None:
        self.batcher.close()


class BatchedOpponentPool:
    def __init__(self, backends: dict[str, Any]) -> None:
        self.backends = backends

    def factory(self, key: str) -> Callable[[], Agent]:
        backend = self.backends[key]
        if isinstance(backend, ActorCriticBatcher):

            def make_actor_critic_agent() -> RolloutAgent:
                agent = RolloutAgent(
                    backend.actor_critic,
                    device="cpu",
                    deterministic=True,
                    inference_backend=backend.submit,
                    record_trajectory=False,
                )
                agent.rollout_batcher = backend
                return agent

            return make_actor_critic_agent
        return backend.make_agent

    def metrics(self) -> dict[str, dict[str, object]]:
        return {name: backend.metrics() for name, backend in self.backends.items()}

    def reset_metrics(self) -> None:
        for backend in self.backends.values():
            backend.reset_metrics()

    def close(self) -> None:
        for backend in self.backends.values():
            backend.close()
