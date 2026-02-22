# app.py
# Drop-in stable control panel server for WhiteNoise Pi
#
# Key improvements vs your current version:
# - All endpoints return JSON (no HTML 500 pages injected into your UI)
# - Track selection is an immediate cutover: hard_stop -> load -> start (no waiting)
# - Pause is a toggle-safe endpoint (pause/resume depending on state)
# - /status reflects the engine’s stable get_state() schema (and stays backward-friendly)
# - Timer + motion endpoints are implemented safely (work even if engine lacks methods)

from __future__ import annotations

from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from wifi_api import bp as wifi_api_bp
import subprocess
import os
from pathlib import Path
from typing import Any

from audio_engine import get_engine

app = Flask(__name__)
app.register_blueprint(wifi_api_bp)
app.secret_key = "littlefoot"

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_EXTS = {".wav"}

ENGINE = get_engine(force_new=True)

LAST_TRACK: str | None = None


# ==============================
# Helpers
# ==============================

def json_error(msg: str, status: int = 400, **extra: Any):
    payload = {"ok": False, "error": msg}
    payload.update(extra)
    return jsonify(payload), status


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

    best: dict[str, dict] = {}
    for n in nets:
        s = n["ssid"]
        if (s not in best) or (n["signal"] > best[s]["signal"]):
            best[s] = n

    return sorted(best.values(), key=lambda x: x["signal"], reverse=True)


def engine_state() -> dict:
    """
    Normalize engine.get_state() into a safe dict even if the engine changes.
    """
    try:
        st = ENGINE.get_state()
        if isinstance(st, dict):
            return st
    except Exception:
        app.logger.exception("ENGINE.get_state failed")
    return {"status": "error", "track": None, "volume": None, "paused": False, "playing": False, "error": "get_state failed"}


# ==============================
# Routes
# ==============================

@app.get("/")
def index():
    # index.html may hardcode track names; keep list_tracks for flexibility.
    return render_template("index.html", tracks=list_tracks(), selected_track=None)


@app.post("/play")
def play():
    """
    IMPORTANT: This endpoint performs an immediate cutover for stability:
      hard_stop -> load -> start
    No fade-out waits or polling. This eliminates multi-second "dead air" delays
    caused by application-side waiting.
    """
    global LAST_TRACK

    data = request.get_json(silent=True) or {}
    track = (data.get("track") or "").strip()

    # Default to a short fade-in; long fades can feel like "nothing is happening"
    fade_in = float(data.get("fade_in_seconds", 0.25))

    try:
        p = resolve_track_filename(track)
        if not p.exists():
            return json_error(f"track not found: {p.name}", 404)

        # Instant transition
        ENGINE.hard_stop()
        ENGINE.load_wav(str(p))
        ENGINE.start(fade_in_seconds=fade_in)

        LAST_TRACK = p.name

        st = engine_state()
        return jsonify({"ok": True, **st, "last_track": LAST_TRACK})

    except ValueError as e:
        return json_error(f"play error: {e}", 400)
    except Exception as e:
        app.logger.exception("Play failed")
        return json_error(f"play error: {type(e).__name__}: {e}", 500)


@app.post("/pause")
def pause():
    """
    Toggle pause for better UX. If playing -> pause. If paused -> resume.
    """
    try:
        st = engine_state()
        if st.get("playing") and not st.get("paused"):
            # Prefer toggle_pause if available
            if hasattr(ENGINE, "toggle_pause"):
                ENGINE.toggle_pause()
            else:
                ENGINE.pause()
        elif st.get("playing") and st.get("paused"):
            if hasattr(ENGINE, "toggle_pause"):
                ENGINE.toggle_pause()
            else:
                ENGINE.resume()

        return jsonify({"ok": True, **engine_state(), "last_track": LAST_TRACK})
    except Exception as e:
        app.logger.exception("Pause failed")
        return json_error(f"pause error: {type(e).__name__}: {e}", 500)


@app.post("/stop")
def stop():
    try:
        ENGINE.hard_stop()
        return jsonify({"ok": True, **engine_state(), "last_track": LAST_TRACK})
    except Exception as e:
        app.logger.exception("Stop failed")
        return json_error(f"stop error: {type(e).__name__}: {e}", 500)


@app.get("/status")
def status():
    """
    Backward-friendly status:
    - New clients can read 'status/playing/paused/track/volume/error'
    - Old clients can read 'is_playing/is_paused'
    """
    st = engine_state()
    return jsonify({
        "ok": True,
        "status": st.get("status"),
        "track": st.get("track") or LAST_TRACK,
        "volume": st.get("volume"),
        "playing": bool(st.get("playing")),
        "paused": bool(st.get("paused")),
        "error": st.get("error"),
        "last_track": LAST_TRACK,
        # legacy keys:
        "is_playing": bool(st.get("playing")),
        "is_paused": bool(st.get("paused")),
    })


@app.post("/volume")
def volume():
    """
    Accepts {value: 0.0-1.0} or {value: 0-100}
    """
    try:
        data = request.get_json(silent=True) or {}
        if "value" not in data:
            return json_error("Missing 'value' for volume", 400)

        value = float(data.get("value"))
        ENGINE.set_volume(value)

        st = engine_state()
        return jsonify({"ok": True, "volume": st.get("volume"), **st})
    except ValueError:
        return json_error("Volume must be a number", 400)
    except Exception as e:
        app.logger.exception("Volume failed")
        return json_error(f"volume error: {type(e).__name__}: {e}", 500)


@app.post("/motion")
def motion():
    """
    Optional: compatible endpoint. If your engine supports set_motion, call it.
    Otherwise, set cfg directly if present.
    """
    try:
        data = request.get_json(silent=True) or {}

        depth = data.get("depth", None)
        period_s = data.get("period_s", None)

        if depth is None and period_s is None:
            # no-op but OK
            return jsonify({"ok": True, **engine_state()})

        # Prefer engine method if present
        if hasattr(ENGINE, "set_motion"):
            kwargs = {}
            if depth is not None:
                kwargs["depth"] = float(depth)
            if period_s is not None:
                kwargs["period_s"] = float(period_s)
            ENGINE.set_motion(**kwargs)
        else:
            # Fallback: set cfg fields directly if available
            if depth is not None and hasattr(ENGINE, "cfg"):
                ENGINE.cfg.drift_depth = float(depth)
            if period_s is not None and hasattr(ENGINE, "cfg"):
                ENGINE.cfg.drift_period_s = float(period_s)

        return jsonify({"ok": True, **engine_state()})
    except Exception as e:
        app.logger.exception("Motion failed")
        return json_error(f"motion error: {type(e).__name__}: {e}", 500)


@app.post("/timer")
def timer():
    """
    Safe timer endpoint:
    If your engine implements play_for/cancel_timer, we use it.
    Otherwise we return a clear JSON error instead of crashing.
    """
    try:
        data = request.get_json(silent=True) or {}
        action = (data.get("action") or "start").lower().strip()

        if action == "cancel":
            if hasattr(ENGINE, "cancel_timer"):
                ENGINE.cancel_timer()
                return jsonify({"ok": True, "status": "timer_canceled", **engine_state()})
            return json_error("Timer cancel not supported by current audio engine", 400)

        minutes = int(data.get("minutes") or 0)
        if minutes <= 0:
            return json_error("No timer duration provided", 400)

        fade_out_seconds = float(data.get("fade_out_seconds", 90))
        seconds = minutes * 60

        if hasattr(ENGINE, "play_for"):
            ENGINE.play_for(seconds=seconds, fade_out_seconds=fade_out_seconds)
            return jsonify({"ok": True, "status": "timer_set", "minutes": minutes, **engine_state()})

        return json_error("Timer not supported by current audio engine", 400)

    except Exception as e:
        app.logger.exception("Timer failed")
        return json_error(f"timer error: {type(e).__name__}: {e}", 500)


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
        use_reloader=is_dev,
    )
