"""Start FaceGuard without a terminal: double-click FaceGuard.command (macOS),
FaceGuard.bat (Windows) or FaceGuard.sh (Linux), which run this file.

First run: creates `.venv/` and installs requirements.txt (the only step that
needs internet, like any install). Every run after that is offline: it
checks the models, starts the app on a free port on this machine only, opens
the browser, and stops the app when this window is closed or Ctrl+C is
pressed.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
APP = ROOT / "ui" / "app.py"
PREFERRED_PORT = 8501
MIN_PYTHON = (3, 10)
START_TIMEOUT_S = 60
STAMP = VENV / ".faceguard-installed"  # requirements.txt mtime recorded after a good install


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def say(msg: str) -> None:
    print(f"[FaceGuard] {msg}", flush=True)


def ensure_environment() -> Path:
    """Create the virtual environment and install requirements if needed."""
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"FaceGuard needs Python {'.'.join(map(str, MIN_PYTHON))} or newer; "
                         f"this is {sys.version.split()[0]}.")
    py = venv_python()
    if not py.exists():
        say("First run: creating a private Python environment (one time)...")
        venv.create(VENV, with_pip=True)
    req = ROOT / "requirements.txt"
    want = str(req.stat().st_mtime_ns)
    if not STAMP.exists() or STAMP.read_text().strip() != want:
        say("Installing components (one time, needs internet)...")
        subprocess.run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", str(req)],
                       check=True)
        STAMP.write_text(want)
    return py


def ensure_models(py: Path) -> None:
    """Restore the committed detector models if they are missing (rarely needed)."""
    missing = [p for p in ("models/face_detection_yunet_2026may.onnx",
                           "models/haarcascade_frontalface_default.xml") if not (ROOT / p).exists()]
    if missing:
        say("Restoring missing model files...")
        subprocess.run([str(py), str(ROOT / "scripts" / "download_model.py")], check=True)


def free_port(preferred: int = PREFERRED_PORT) -> int:
    """The preferred port if it is free on localhost, otherwise any free one."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free local port.")


def app_command(py: Path, port: int) -> list[str]:
    """Streamlit bound to this machine only; .streamlit/config.toml sets the rest."""
    return [str(py), "-m", "streamlit", "run", str(APP), "--server.port", str(port),
            "--server.address", "localhost", "--server.headless", "true",
            "--browser.gatherUsageStats", "false"]


def wait_until_up(port: int, proc: subprocess.Popen, timeout: float = START_TIMEOUT_S) -> bool:
    """Poll Streamlit's local health endpoint until it answers."""
    url = f"http://localhost:{port}/_stcore/health"
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.4)
    return False


def main() -> int:
    os.chdir(ROOT)  # .streamlit/config.toml is read from the working directory
    py = ensure_environment()
    ensure_models(py)
    port = free_port()
    say("Starting...")
    proc = subprocess.Popen(app_command(py, port))
    try:
        if not wait_until_up(port, proc):
            say("The app did not start. See the messages above.")
            proc.terminate()
            return 1
        url = f"http://localhost:{port}"
        say(f"Running at {url} (this computer only). Close this window or press Ctrl+C to quit.")
        webbrowser.open(url)
        return proc.wait()
    except KeyboardInterrupt:
        say("Stopping...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
