from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from wifi_api import bp as wifi_api_bp
import subprocess
import os
import signal
import threading
import time
from pathlib import Path


app = Flask(__name__)
app.register_blueprint(wifi_api_bp)
app.secret_key = "littlefoot"

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_EXTS = {".wav"}

LAST_TRACK: str | None = None

# --- process-based audio state ---
AUDIO_LOCK = threading.Lock()
AUDIO_PROC: subprocess.Popen | None = None
TIMER_THREAD: threading.Thread | None = None
TIMER_CANCEL = threading.Event()


# ==============================
# Track Helpers
# ==============================

def list_tracks() -> list[str]:
    if not AUDIO_DIR.exists():
        return []
    files = [
        p.name
        for p in AUDIO_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]
    return sorted(files, key=str.lower)


def resolve_track_filename(filename: str) -> Path:
    if not filename:
        raise ValueError("missing track")
    if "/" in filename or "\\" in filename:
        raise ValueError("invalid track filename")

    p = (AUDIO_DIR / filename).resolve()
    if AUDIO_DIR.resolve() not in p.parents:
        raise ValueError("invalid track path")
    return p


# ==============================
# Wi-Fi Utilities
# ==============================

def nmcli(args: list[str]) -> str:
    cmd = ["sudo", "/usr/bin/nmcli"] + args
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def scan_wifi() -> list[dict]:
    nmcli(["dev", "wifi", "rescan"])
    raw = nmcli(["-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list"])

    nets = []
    for line in raw.splitlines():
        parts = line.split(":")
        if len(parts) < 3:
            continue
        ssid, signal, security = parts[0].strip(), parts[1], parts[2]
        if not ssid:
            continue
        try:
            signal_i = int(signal) if signal else 0
        except ValueError:
            signal_i = 0
        nets.append({"ssid": ssid, "signal": signal_i, "security": security or ""})

    best = {}
    for n in nets:
        s = n["ssid"]
        if (s not in best) or (n["signal"] > best[s]["signal"]):
            best[s] = n

    return sorted(best.values(), key=lambda x: x["signal"], reverse=True)


# ==============================
# Audio Process Control
# ==============================

def _kill_proc(proc: subprocess.Popen):
    """Best-effort terminate/kill of the audio subprocess."""
    try:
        # terminate process group if we started one
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.terminate()
    except Exception:
        pass

    # give it a moment
    try:
        proc.wait(timeout=0.8)
        return
    except Exception:
        pass

    # hard kill
    try:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
    except Exception:
        pass

    try:
        proc.wait(timeout=0.8)
    except Exception:
        pass


def stop_audio():
    global AUDIO_PROC
    with AUDIO_LOCK:
        if AUDIO_PROC is None:
            app.logger.warning("[AUDIODBG] stop_audio: no active proc")
            return
        proc = AUDIO_PROC
        AUDIO_PROC = None

    app.logger.warning(f"[AUDIODBG] stop_audio: terminating pid={proc.pid}")
    _kill_proc(proc)
    app.logger.warning(f"[AUDIODBG] stop_audio: terminated pid={proc.pid} rc={proc.returncode}")


def start_audio_wav(path: Path):
    """
    Start WAV playback via aplay as a separate process (non-blocking).
    Logs stderr/stdout and exit codes so we can debug ALSA issues.
    """
    global AUDIO_PROC

    app.logger.warning(f"[AUDIODBG] start_audio_wav requested path={path}")
    stop_audio()

    proc = subprocess.Popen(
        ["aplay", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    with AUDIO_LOCK:
        AUDIO_PROC = proc

    app.logger.warning(f"[AUDIODBG] aplay started pid={proc.pid}")

    def drain_and_log(p: subprocess.Popen):
        try:
            out, err = p.communicate()
        except Exception as e:
            app.logger.error(f"[AUDIODBG] aplay communicate failed: {e}")
            return

        # Keep logs readable (last 800 chars)
        out_tail = (out or "")[-800:]
        err_tail = (err or "")[-800:]

        app.logger.warning(
            f"[AUDIODBG] aplay exited pid={p.pid} rc={p.returncode} "
            f"stdout_tail={out_tail!r} stderr_tail={err_tail!r}"
        )

    threading.Thread(target=drain_and_log, args=(proc,), daemon=True).start()


def is_playing() -> bool:
    with AUDIO_LOCK:
        proc = AUDIO_PROC
    if proc is None:
        return False
    return proc.poll() is None


def cancel_timer():
    TIMER_CANCEL.set()


def _timer_worker(seconds: int):
    TIMER_CANCEL.clear()
    start = time.time()
    while True:
        if TIMER_CANCEL.is_set():
            return
        if time.time() - start >= seconds:
            break
        time.sleep(0.25)
    stop_audio()


# ==============================
# Routes
# ==============================

@app.route("/")
def index():
    return render_template("index.html", tracks=list_tracks(), selected_track=None)


@app.route("/play", methods=["POST"])
def play():
    global LAST_TRACK

    data = request.get_json(silent=True) or {}
    track = data.get("track")

    # fade_in_seconds accepted but not used with aplay (kept for API compatibility)
    # fade_in = float(data.get("fade_in_seconds", 10))

    try:
        p = resolve_track_filename(track)
        if not p.exists():
            return f"track not found: {p.name}", 404

        start_audio_wav(p)
        LAST_TRACK = p.name

        return jsonify({"ok": True, "status": "playing", "track": p.name})

    except ValueError as e:
        return f"play error: {e}", 400
    except Exception as e:
        app.logger.exception("Play failed")
        return f"play error: {type(e).__name__}: {e}", 500


@app.route("/pause", methods=["POST"])
def pause():
    # Treat pause as stop (reliable + matches your current UI behavior)
    stop_audio()
    return jsonify({"ok": True, "status": "paused"})


@app.route("/stop", methods=["POST"])
def stop():
    stop_audio()
    return jsonify({"ok": True, "status": "stopped"})


@app.route("/timer", methods=["POST"])
def timer():
    global TIMER_THREAD

    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "start").lower().strip()

    if action == "cancel":
        cancel_timer()
        return jsonify({"ok": True, "status": "timer_canceled"})

    minutes = int(data.get("minutes") or 0)
    if minutes <= 0:
        return jsonify({"error": "No timer duration provided"}), 400

    seconds = minutes * 60

    # (Re)start timer thread
    cancel_timer()
    TIMER_THREAD = threading.Thread(target=_timer_worker, args=(seconds,), daemon=True)
    TIMER_THREAD.start()

    return jsonify({"ok": True, "status": "timer_set", "minutes": minutes})


@app.route("/status")
def status():
    return jsonify({
        "ok": True,
        "is_playing": is_playing(),
        "last_track": LAST_TRACK,
    })


@app.route("/volume", methods=["POST"])
def volume():
    # Not supported in this aplay-backed mode (easy to add via amixer later)
    return jsonify({"ok": False, "error": "volume control not supported in aplay mode"}), 400


@app.route("/motion", methods=["POST"])
def motion():
    # Not supported in this aplay-backed mode
    return jsonify({"ok": False, "error": "motion control not supported in aplay mode"}), 400


@app.route("/setup", methods=["GET", "POST"])
def setup_wifi():
    if request.method == "POST":
        ssid = (request.form.get("ssid") or "").strip()
        password = (request.form.get("password") or "").strip()

        if not ssid:
            flash("Please select a Wi-Fi network (SSID).", "error")
            return redirect(url_for("setup_wifi"))

        try:
            if password:
                nmcli(["dev", "wifi", "connect", ssid, "password", password])
            else:
                nmcli(["dev", "wifi", "connect", ssid])

            try:
                nmcli(["con", "down", "transmogrifier-hotspot"])
            except Exception:
                pass

            flash(f"Connected to {ssid}.", "success")
            return redirect(url_for("index"))

        except subprocess.CalledProcessError:
            flash("Failed to connect. Check password.", "error")
            return redirect(url_for("setup_wifi"))

    try:
        networks = scan_wifi()
    except Exception:
        networks = []

    return render_template("setup.html", networks=networks)


if __name__ == "__main__":
    mode = os.getenv("APP_MODE", "appliance").lower()
    is_dev = (mode == "dev")

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=is_dev,
        use_reloader=is_dev
    )
