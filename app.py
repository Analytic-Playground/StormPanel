from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from wifi_api import bp as wifi_api_bp
import subprocess
import os
from pathlib import Path

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
# Routes
# ==============================

@app.route("/")
def index():
    # You’re hardcoding track names in index.html now; this is harmless to keep.
    return render_template("index.html", tracks=list_tracks(), selected_track=None)


@app.route("/play", methods=["POST"])
def play():
    global LAST_TRACK

    data = request.get_json(silent=True) or {}
    track = data.get("track")
    fade_in = float(data.get("fade_in_seconds", 5))

    try:
        p = resolve_track_filename(track)
        if not p.exists():
            return f"track not found: {p.name}", 404

        ENGINE.load_wav(str(p))
        ENGINE.start(fade_in_seconds=fade_in)
        LAST_TRACK = p.name

        return jsonify({"ok": True, "status": "playing", "track": p.name})

    except ValueError as e:
        return f"play error: {e}", 400
    except Exception as e:
        app.logger.exception("Play failed")
        return f"play error: {type(e).__name__}: {e}", 500


@app.route("/pause", methods=["POST"])
def pause():
    ENGINE.pause()
    return jsonify({"ok": True, "status": "paused"})


@app.route("/stop", methods=["POST"])
def stop():
    ENGINE.hard_stop()
    return jsonify({"ok": True, "status": "stopped"})


@app.route("/timer", methods=["POST"])
def timer():
    data = request.get_json(silent=True) or {}
    action = (data.get("action") or "start").lower().strip()

    if action == "cancel":
        ENGINE.cancel_timer()
        return jsonify({"ok": True, "status": "timer_canceled"})

    minutes = int(data.get("minutes") or 0)
    if minutes <= 0:
        return jsonify({"error": "No timer duration provided"}), 400

    fade_out_seconds = float(data.get("fade_out_seconds", 90))
    seconds = minutes * 60

    ENGINE.play_for(seconds=seconds, fade_out_seconds=fade_out_seconds)
    return jsonify({"ok": True, "status": "timer_set", "minutes": minutes})


@app.route("/status")
def status():
    st = ENGINE.get_state()
    return jsonify({
        "ok": True,
        "is_playing": bool(st.get("is_playing")),
        "is_paused": bool(st.get("is_paused")),
        "fade_state": st.get("fade_state"),
        "has_stream": bool(st.get("has_stream")),
        "last_track": LAST_TRACK,
        "sr": st.get("sr"),
    })


@app.route("/volume", methods=["POST"])
def volume():
    data = request.get_json(silent=True) or {}
    value = float(data.get("value", 0.7))
    ENGINE.set_volume(value)
    return jsonify({"ok": True, "volume": value})


@app.route("/motion", methods=["POST"])
def motion():
    # Optional: wire this later from UI. For now keep endpoint compatible.
    data = request.get_json(silent=True) or {}
    depth = float(data.get("depth", ENGINE.cfg.drift_depth))
    period_s = float(data.get("period_s", ENGINE.cfg.drift_period_s))
    ENGINE.set_motion(depth=depth, period_s=period_s)
    return jsonify({"ok": True, "depth": ENGINE.cfg.drift_depth, "period_s": ENGINE.cfg.drift_period_s})


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
