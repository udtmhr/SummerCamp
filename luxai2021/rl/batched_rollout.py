from __future__ import annotations

# ruff: noqa: BLE001
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import torch
from torch import Tensor

from luxai2021.env.agent import Agent
from luxai2021.imitation.agent import BehaviorCloningAgent, FirstPlaceAgent
from luxai2021.imitation.first_place import predict_first_place
from luxai2021.imitation.schema import snapshot_from_game
from luxai2021.rl.policy import FullTurnActorCritic, RolloutAgent

if TYPE_CHECKING:
    from collections.abc import Callable

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


@dataclass
class _Pending(Generic[RequestT, ResponseT]):
    payload: RequestT
    ready: threading.Event
    result: ResponseT | None = None
    error: BaseException | None = None


class InferenceBatcher(Generic[RequestT, ResponseT]):
    """Collect synchronous requests from environment threads into GPU batches."""

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
        self.requests: queue.Queue[_Pending[RequestT, ResponseT] | None] = queue.Queue()
        self.batch_sizes: list[int] = []
        self.inference_seconds = 0.0
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, payload: RequestT) -> ResponseT:
        pending: _Pending[RequestT, ResponseT] = _Pending(payload, threading.Event())
        self.requests.put(pending)
        pending.ready.wait()
        if pending.error is not None:
            raise RuntimeError("Batched inference failed") from pending.error
        return pending.result  # type: ignore[return-value]

    @staticmethod
    def _validate_results(results: list[ResponseT], expected: int) -> None:
        if len(results) != expected:
            raise RuntimeError("Batched inference returned the wrong number of results")

    def _run(self) -> None:  # noqa: C901
        while True:
            first = self.requests.get()
            if first is None:
                return
            pending = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(pending) < self.max_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    request = self.requests.get(timeout=timeout)
                except queue.Empty:
                    break
                if request is None:
                    self.requests.put(None)
                    break
                pending.append(request)
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
                for request in pending:
                    request.ready.set()

    def metrics(self) -> dict[str, float]:
        count = len(self.batch_sizes)
        return {
            "batches": float(count),
            "samples": float(sum(self.batch_sizes)),
            "mean_batch_size": float(sum(self.batch_sizes) / count) if count else 0.0,
            "max_batch_size": float(max(self.batch_sizes, default=0)),
            "inference_seconds": self.inference_seconds,
        }

    def close(self) -> None:
        self.requests.put(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("Inference batcher did not stop")


def _cpu_outputs(output: dict[str, Tensor], index: int) -> dict[str, Tensor]:
    return {name: logits[index : index + 1].float().cpu() for name, logits in output.items()}


class ActorCriticBatcher:
    def __init__(
        self, actor_critic: FullTurnActorCritic, device: torch.device, max_batch_size: int, *, name: str
    ) -> None:
        self.actor_critic = actor_critic
        self.device = device

        def infer(observations: list[Tensor]) -> list[tuple[dict[str, Tensor], Tensor]]:
            batch = torch.stack(observations).to(device, non_blocking=True)
            if device.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                output, values = actor_critic(batch)
            return [(_cpu_outputs(output, index), values[index].float().cpu()) for index in range(len(observations))]

        self.batcher = InferenceBatcher(infer, max_batch_size=max_batch_size, name=name)

    def submit(self, observation: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        return self.batcher.submit(observation)

    def metrics(self) -> dict[str, float]:
        return self.batcher.metrics()

    def close(self) -> None:
        self.batcher.close()


class BehaviorCloningBatcher:
    def __init__(self, prototype: BehaviorCloningAgent, max_batch_size: int, *, name: str) -> None:
        self.prototype = prototype

        def infer(observations: list[Tensor]) -> list[dict[str, Tensor]]:
            batch = torch.stack(observations).to(prototype.device, non_blocking=True)
            if prototype.device.type == "cuda":
                batch = batch.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():
                output = prototype._predict(batch)  # noqa: SLF001
            return [_cpu_outputs(output, index) for index in range(len(observations))]

        self.batcher = InferenceBatcher(infer, max_batch_size=max_batch_size, name=name)

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

            def _predict(self, observation: Tensor) -> dict[str, Tensor]:
                return submit(observation[0].cpu())

        return BrokeredBehaviorCloningAgent()

    def metrics(self) -> dict[str, float]:
        return self.batcher.metrics()

    def close(self) -> None:
        self.batcher.close()


class FirstPlaceBatcher:
    def __init__(self, prototype: FirstPlaceAgent, max_batch_size: int, *, name: str) -> None:
        self.prototype = prototype

        def infer(snapshots: list[Any]) -> list[dict[str, Tensor]]:
            output = predict_first_place(
                prototype.model,
                snapshots,
                device=prototype.device,
                rot180=prototype.tta == "rot180",
            )
            return [_cpu_outputs(output, index) for index in range(len(snapshots))]

        self.batcher = InferenceBatcher(infer, max_batch_size=max_batch_size, name=name)

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

            def process_turn(self, game: object, team: int) -> list[object]:
                snapshot = snapshot_from_game(game)
                output = submit(snapshot)
                selected_output = self._select_team_output(output, team)
                unit_choices = self._choose_units(game, team, snapshot, selected_output, 0, 0)
                actions = self._choices_to_actions(unit_choices, team)
                actions.extend(self._city_actions(game, team, snapshot, selected_output, 0, 0))
                return actions

        return BrokeredFirstPlaceAgent()

    def metrics(self) -> dict[str, float]:
        return self.batcher.metrics()

    def close(self) -> None:
        self.batcher.close()


class BatchedOpponentPool:
    def __init__(self, backends: dict[str, Any]) -> None:
        self.backends = backends

    def factory(self, key: str) -> Callable[[], Agent]:
        backend = self.backends[key]
        if isinstance(backend, ActorCriticBatcher):
            return lambda: RolloutAgent(
                backend.actor_critic,
                device="cpu",
                deterministic=True,
                inference_backend=backend.submit,
                record_trajectory=False,
            )
        return backend.make_agent

    def metrics(self) -> dict[str, dict[str, float]]:
        return {name: backend.metrics() for name, backend in self.backends.items()}

    def close(self) -> None:
        for backend in self.backends.values():
            backend.close()
