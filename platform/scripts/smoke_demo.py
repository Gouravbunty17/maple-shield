"""Bounded local smoke test for the Maple Shield demo stack.

This script starts command-api and fusion-engine on random local ports, runs
the mock edge-agent for a short burst, verifies the alert path, and then
stops everything. It is intentionally local-only and uses no model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import httpx


ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        str(ROOT),
        str(ROOT / "command-api"),
        str(ROOT / "edge-agent"),
        str(ROOT / "fusion-engine"),
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    env["MAPLE_SHIELD_LAWFUL_USE_ACK"] = "true"
    if extra:
        env.update(extra)
    return env


def _start(name: str, cwd: Path, args: list[str], log_dir: Path, env: dict[str, str]):
    stdout = (log_dir / f"{name}.out.log").open("w", encoding="utf-8")
    stderr = (log_dir / f"{name}.err.log").open("w", encoding="utf-8")
    try:
        return subprocess.Popen(
            args,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    finally:
        stdout.close()
        stderr.close()


def _wait_json(url: str, *, timeout_s: float = 10.0) -> dict:
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.time() < deadline:
            try:
                response = client.get(url)
                if 200 <= response.status_code < 300:
                    return response.json()
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for {url}")


def _fetch_json(url: str):
    with httpx.Client(timeout=3.0, trust_env=False) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _terminate(processes: Iterable[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 5
    for proc in processes:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            proc.kill()


def run_smoke(frames: int, fps: float, log_dir: Path) -> dict:
    command_port = _free_port()
    fusion_port = _free_port()
    command_url = f"http://127.0.0.1:{command_port}"
    fusion_url = f"http://127.0.0.1:{fusion_port}"

    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    db_file = Path(tempfile.gettempdir()) / f"maple-shield-smoke-{os.getpid()}.sqlite3"
    if db_file.exists():
        db_file.unlink()

    try:
        command_env = _env({"MAPLE_SHIELD_DB": str(db_file)})
        processes.append(
            _start(
                "command-api",
                ROOT / "command-api",
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(command_port)],
                log_dir,
                command_env,
            )
        )
        _wait_json(f"{command_url}/healthz")

        fusion_env = _env({"MAPLE_SHIELD_COMMAND_API": command_url})
        processes.append(
            _start(
                "fusion-engine",
                ROOT,
                [sys.executable, "-m", "uvicorn", "fusion.main:app", "--host", "127.0.0.1", "--port", str(fusion_port)],
                log_dir,
                fusion_env,
            )
        )
        _wait_json(f"{fusion_url}/healthz")

        edge = subprocess.run(
            [
                sys.executable,
                "-m",
                "edge_agent.main",
                "--source",
                "mock",
                "--detector",
                "mock",
                "--n-frames",
                str(frames),
                "--fps",
                str(fps),
                "--fusion",
                fusion_url,
            ],
            cwd=str(ROOT / "edge-agent"),
            env=_env(),
            capture_output=True,
            text=True,
            timeout=max(10.0, frames / max(0.1, fps) + 10.0),
        )
        (log_dir / "edge-agent.out.log").write_text(edge.stdout, encoding="utf-8")
        (log_dir / "edge-agent.err.log").write_text(edge.stderr, encoding="utf-8")
        if edge.returncode != 0:
            raise RuntimeError(f"edge-agent failed with exit code {edge.returncode}")

        alerts = _fetch_json(f"{command_url}/alerts")
        incidents = _fetch_json(f"{command_url}/incidents")
        audit = _fetch_json(f"{command_url}/audit")
        tracks = _fetch_json(f"{fusion_url}/tracks")

        if not alerts:
            raise RuntimeError("smoke produced no alerts")
        if not incidents:
            raise RuntimeError("smoke produced no incidents")
        if not audit.get("verified"):
            raise RuntimeError("audit chain did not verify")

        return {
            "status": "ok",
            "command_api": command_url,
            "fusion_engine": fusion_url,
            "frames": frames,
            "alerts": len(alerts),
            "incidents": len(incidents),
            "tracks": len(tracks),
            "audit_verified": bool(audit.get("verified")),
            "log_dir": str(log_dir),
        }
    finally:
        _terminate(processes)
        try:
            db_file.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--log-dir", default=str(ROOT / ".demo-logs" / "smoke"))
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)

    result = run_smoke(args.frames, args.fps, Path(args.log_dir))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            "[smoke] ok "
            f"alerts={result['alerts']} incidents={result['incidents']} "
            f"tracks={result['tracks']} logs={result['log_dir']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
