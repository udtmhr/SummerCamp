from __future__ import annotations

# ruff: noqa: S310
import base64
import hmac
import io
import json
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

from luxai2021.rl.evolution import (
    CandidateResult,
    EvolutionJob,
    EvolutionStore,
    FilesystemJobQueue,
)

JOB_API_VERSION = 1


def encode_artifact_directory(path: Path) -> str | None:
    """Return a zip archive suitable for JSON transport, or None when absent."""
    if not path.exists():
        return None
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in sorted(item for item in path.rglob("*") if item.is_file()):
            archive.write(artifact, artifact.relative_to(path).as_posix())
    return base64.b64encode(output.getvalue()).decode("ascii")


def extract_artifact_directory(encoded: str, destination: Path) -> None:
    payload = base64.b64decode(encoded, validate=True)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Artifact archive contains an unsafe path")
            if info.is_dir():
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


class JobApiClient:
    def __init__(self, base_url: str, *, token: str | None = None, timeout_seconds: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload, separators=(",", ":")).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status == HTTPStatus.NO_CONTENT:
                    return None
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            message = f"Job API returned HTTP {error.code}: {detail}"
            raise RuntimeError(message) from error

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        return self._post("/v1/claim", {"worker_id": worker_id})

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.base_url}/healthz", method="GET")
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read())

    def complete(
        self,
        *,
        lease_id: str,
        job: EvolutionJob,
        result: CandidateResult,
        artifact_dir: Path,
    ) -> None:
        response = self._post(
            "/v1/complete",
            {
                "lease_id": lease_id,
                "job_id": job.job_id,
                "result": asdict(result),
                "artifacts_zip_base64": encode_artifact_directory(artifact_dir),
            },
        )
        if response is None or not response.get("ok"):
            raise RuntimeError("Job API did not acknowledge completion")


class _JobApiHttpServer(ThreadingHTTPServer):
    daemon_threads = True


class JobApiServer:
    """Expose a coordinator-owned filesystem queue through a small JSON API."""

    def __init__(
        self,
        listen: str,
        *,
        run_dir: Path,
        queue: FilesystemJobQueue,
        token: str | None = None,
    ) -> None:
        host, separator, port_text = listen.rpartition(":")
        if not separator or not host:
            raise ValueError("Job API listen address must use HOST:PORT")
        self.run_dir = run_dir
        self.store = EvolutionStore(run_dir)
        self.queue = queue
        self.token = token
        self.httpd = _JobApiHttpServer((host, int(port_text)), self._handler_class())
        self.thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:  # noqa: C901
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "LuxEvolutionJobAPI/1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _json(self, status: HTTPStatus, payload: dict[str, Any] | None = None) -> None:
                body = b"" if payload is None else json.dumps(payload, separators=(",", ":")).encode()
                self.send_response(status)
                if body:
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _authorized(self) -> bool:
                if owner.token is None:
                    return True
                provided = self.headers.get("Authorization", "").removeprefix("Bearer ")
                return hmac.compare_digest(provided, owner.token)

            def do_GET(self) -> None:
                if self.path in {"/", "/healthz"}:
                    self._json(HTTPStatus.OK, {"status": "ok", "api_version": JOB_API_VERSION})
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:
                if not self._authorized():
                    self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length)) if length else {}
                    if self.path == "/v1/claim":
                        self._claim(payload)
                    elif self.path == "/v1/complete":
                        self._complete(payload)
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                except (KeyError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

            def _claim(self, payload: dict[str, Any]) -> None:
                worker_id = str(payload["worker_id"])
                claimed = owner.queue.claim(worker_id)
                if claimed is None:
                    self._json(HTTPStatus.NO_CONTENT)
                    return
                job, claimed_path = claimed
                candidates = owner.store.candidates()
                candidate = next(item for item in candidates if item.candidate_id == job.candidate_id)
                manifest = json.loads((owner.run_dir / "manifest.json").read_text(encoding="utf-8"))
                input_artifact = None
                if job.stage == "medium-resattn8":
                    input_stage = "short-resattn8"
                    input_base = "resattn8"
                    input_dir = owner.run_dir / "artifacts" / job.candidate_id / input_stage / input_base
                    encoded = encode_artifact_directory(input_dir)
                    if encoded is None:
                        raise ValueError("Medium job is missing its short-stage checkpoint")
                    input_artifact = {
                        "stage": input_stage,
                        "base_name": input_base,
                        "zip_base64": encoded,
                    }
                self._json(
                    HTTPStatus.OK,
                    {
                        "api_version": JOB_API_VERSION,
                        "lease_id": claimed_path.name,
                        "job": job.to_dict(),
                        "candidate": candidate.to_dict(),
                        "candidates": [item.to_dict() for item in candidates],
                        "results": [asdict(item) for item in owner.store.results()],
                        "manifest": manifest,
                        "input_artifact": input_artifact,
                    },
                )

            def _complete(self, payload: dict[str, Any]) -> None:
                lease_id = str(payload["lease_id"])
                if Path(lease_id).name != lease_id:
                    raise ValueError("Invalid lease id")
                claimed_path = owner.queue.running_dir / lease_id
                job_id = str(payload["job_id"])
                if not claimed_path.exists():
                    completed_path = owner.queue.completed_dir / f"{job_id}.json"
                    if completed_path.exists():
                        self._json(HTTPStatus.OK, {"ok": True, "duplicate": True})
                        return
                    raise ValueError("Unknown or expired lease")
                job = EvolutionJob.from_dict(json.loads(claimed_path.read_text(encoding="utf-8")))
                if job.job_id != job_id:
                    raise ValueError("Lease and job id do not match")
                result = CandidateResult(**payload["result"])
                if result.candidate_id != job.candidate_id or result.stage != job.stage:
                    raise ValueError("Result does not match claimed job")
                encoded = payload.get("artifacts_zip_base64")
                artifact_dir = owner.run_dir / "artifacts" / job.candidate_id / job.stage / job.base_name
                if encoded:
                    extract_artifact_directory(str(encoded), artifact_dir)
                metrics = dict(result.metrics)
                if result.status == "completed":
                    metrics["checkpoint"] = str(artifact_dir / "best.pt")
                    result = CandidateResult(
                        result.candidate_id,
                        result.stage,
                        result.status,
                        result.score_rate,
                        result.teacher_score_rate,
                        result.kl,
                        result.duration_seconds,
                        metrics,
                        result.error,
                    )
                owner.store.save_result(result)
                owner.queue.complete(claimed_path, result)
                self._json(HTTPStatus.OK, {"ok": True})

        return Handler

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="lux-job-api", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None
