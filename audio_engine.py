# audio_engine.py
#
# Stable, low-latency, stream-persistent audio engine for Raspberry Pi.
# - Keeps ONE OutputStream alive (prevents multi-second start delays on some devices)
# - Never raises out of the audio callback (prevents "plays once then dead")
# - Supports: load_wav, start, stop (fade), hard_stop, pause/resume/toggle_pause,
#            set_volume (0-1 or 0-100), get_state, play_for, cancel_timer
#
# Drop-in compatible with your Flask control panel.

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class MotionConfig:
    volume: float = 0.65          # 0.0-1.0
    drift_depth: float = 0.05     # 0.0 disables
    drift_period_s: float = 80.0
    fade_in_s: float = 120.0      # 2 minutes fade in
    fade_out_s: float = 360.0     # 6 minutes fade out


class AudioEngine:
    """
    Persistent-stream loop player.
    Stream remains open; "stop/pause" outputs silence.
    """

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self.cfg = MotionConfig()

        self._audio: Optional[np.ndarray] = None  # float32, shape (n,2)
        self._sr: int = 44100
        self._pos: int = 0

        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.RLock()

        self._is_playing: bool = False
        self._is_paused: bool = False

        self._fade_state: str = "none"  # "none" | "in" | "out"
        self._fade_t0: float = 0.0
        self._t0: float = time.monotonic()

        # UI/reporting
        self.current_track: Optional[str] = None
        self.status: str = "stopped"  # "playing" | "paused" | "stopped" | "error"
        self.last_error: Optional[str] = None

        # Sleep timer
        self._timer_cancel: Optional[threading.Event] = None

    # ------------------------
    # Public API
    # ------------------------

    def load_wav(self, path: str):
        """
        Load audio into memory as float32 stereo, reset position.
        """
        data, sr = sf.read(path, dtype="float32", always_2d=True)

        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]

        data = np.ascontiguousarray(data, dtype=np.float32)

        with self._lock:
            self._audio = data
            self._sr = int(sr)
            self._pos = 0
            self.current_track = path
            self.last_error = None
            if self.status == "error":
                self.status = "stopped"

        # Ensure stream exists (but doesn't start "playing" until start())
        self._ensure_stream()

    def set_volume(self, volume: float):
        """
        Accepts 0.0-1.0 or 0-100.
        """
        v = float(volume)
        if v > 1.0:
            v = v / 100.0
        v = max(0.0, min(1.0, v))
        with self._lock:
            self.cfg.volume = v

    def start(self, fade_in_seconds: Optional[float] = None):
        """
        Begin playback of loaded audio (looping). Stream stays alive.
        """
        with self._lock:
            if self._audio is None:
                raise RuntimeError("No audio loaded.")

            if fade_in_seconds is not None:
                self.cfg.fade_in_s = max(0.0, float(fade_in_seconds))

            self._is_playing = True
            self._is_paused = False
            self.status = "playing"
            self.last_error = None

            self._fade_state = "in" if self.cfg.fade_in_s > 0 else "none"
            self._fade_t0 = time.monotonic()
            self._t0 = time.monotonic()
            self._pos = 0

        self._ensure_stream()

    def stop(self, fade_out_seconds: Optional[float] = None):
        """
        Fade out to silence and then mark stopped (stream remains alive).
        """
        with self._lock:
            if fade_out_seconds is not None:
                self.cfg.fade_out_s = max(0.0, float(fade_out_seconds))

            if not self._is_playing:
                self.status = "stopped"
                self._fade_state = "none"
                return

            self._fade_state = "out"
            self._fade_t0 = time.monotonic()
            # status stays "playing" during fade-out for UI
            self.status = "paused" if self._is_paused else "playing"

    def hard_stop(self):
        """
        Immediate silence + reset, keep stream alive.
        """
        with self._lock:
            self._is_playing = False
            self._is_paused = False
            self._fade_state = "none"
            self._pos = 0
            self.status = "stopped"

    def pause(self):
        with self._lock:
            if self._is_playing:
                self._is_paused = True
                self.status = "paused"

    def resume(self):
        with self._lock:
            if self._is_playing:
                self._is_paused = False
                self.status = "playing"

    def toggle_pause(self):
        with self._lock:
            if not self._is_playing:
                return
            if self._is_paused:
                self._is_paused = False
                self.status = "playing"
            else:
                self._is_paused = True
                self.status = "paused"

    def play_for(self, seconds: float, fade_out_seconds: float = 360.0):
        """
        Play for a set duration, then fade out and stop.
        Cancels any existing timer before starting a new one.
        """
        self.cancel_timer()

        fade = max(0.0, float(fade_out_seconds))
        wait = max(0.0, float(seconds) - fade)

        self._timer_cancel = threading.Event()
        cancel_event = self._timer_cancel

        def _run():
            if cancel_event.wait(timeout=wait):
                return  # canceled
            if not cancel_event.is_set():
                self.stop(fade_out_seconds=fade)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def cancel_timer(self):
        """
        Cancel any running sleep timer.
        """
        event = getattr(self, "_timer_cancel", None)
        if event is not None:
            event.set()
        self._timer_cancel = None

    def close(self):
        """
        Stop and close stream (only use on shutdown / engine recreation).
        """
        with self._lock:
            self._is_playing = False
            self._is_paused = False
            self._fade_state = "none"
            self._pos = 0
            self.status = "stopped"
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "track": self.current_track,
                "volume": float(self.cfg.volume),
                "paused": bool(self._is_paused),
                "playing": bool(self._is_playing),
                "error": self.last_error,
            }

    # ------------------------
    # Internal
    # ------------------------

    def _ensure_stream(self):
        """
        Create/start a stream once. If sample rate changes, recreate cleanly.
        """
        with self._lock:
            sr = self._sr
            device = self.device
            stream = self._stream

        # If no stream, create it
        if stream is None:
            self._create_stream(sr, device)
            return

        # If stream exists but inactive, attempt restart
        if not getattr(stream, "active", False):
            try:
                stream.start()
            except Exception:
                # recreate
                self._create_stream(sr, device)
            return

        # Stream active; ensure SR matches (PortAudio doesn't allow changing SR)
        # If SR differs, recreate.
        try:
            current_sr = getattr(stream, "samplerate", sr)
        except Exception:
            current_sr = sr

        if int(current_sr) != int(sr):
            self._create_stream(sr, device)

    def _create_stream(self, sr: int, device: Optional[int]):
        """
        Stop/close existing stream and create a new one with the given SR.
        """
        with self._lock:
            old = self._stream
            self._stream = None

        if old is not None:
            try:
                old.stop()
            except Exception:
                pass
            try:
                old.close()
            except Exception:
                pass

        try:
            new_stream = sd.OutputStream(
                samplerate=int(sr),
                channels=2,
                dtype="float32",
                callback=self._callback,
                device=device,
                blocksize=0,
            )
            new_stream.start()
        except Exception as e:
            with self._lock:
                self.last_error = f"Stream start failed: {e}"
                self.status = "error"
                self._is_playing = False
                self._is_paused = False
                self._fade_state = "none"
                self._stream = None
            raise

        with self._lock:
            self._stream = new_stream

    def _current_fade_gain(self) -> float:
        now = time.monotonic()

        if self._fade_state == "in":
            dur = max(0.001, self.cfg.fade_in_s)
            x = min(1.0, (now - self._fade_t0) / dur)
            if x >= 1.0:
                self._fade_state = "none"
            return 0.5 - 0.5 * math.cos(math.pi * x)

        if self._fade_state == "out":
            dur = max(0.001, self.cfg.fade_out_s)
            x = min(1.0, (now - self._fade_t0) / dur)
            g = 1.0 - x
            return max(0.0, g)

        return 1.0

    def _callback(self, outdata, frames, time_info, status):
        """
        PortAudio callback: MUST NEVER raise.
        """
        try:
            with self._lock:
                # Silence if stopped/paused/unloaded
                if (not self._is_playing) or self._is_paused or (self._audio is None):
                    outdata[:] = 0
                    return

                audio = self._audio
                n = audio.shape[0]
                if n <= 0:
                    outdata[:] = 0
                    return

                end = self._pos + frames
                if end <= n:
                    chunk = audio[self._pos:end]
                    self._pos = 0 if end == n else end
                else:
                    first = audio[self._pos:n]
                    rest = audio[0:(end - n)]
                    chunk = np.vstack((first, rest))
                    self._pos = end - n

                # Drift
                t = time.monotonic() - self._t0
                depth = self.cfg.drift_depth
                period = self.cfg.drift_period_s
                s = math.sin(2.0 * math.pi * (t / period)) if depth > 0 else 0.0
                gL = 1.0 + depth * s
                gR = 1.0 - depth * s

                fade = self._current_fade_gain()
                vol = self.cfg.volume * fade

                outdata[:, 0] = chunk[:, 0] * (vol * gL)
                outdata[:, 1] = chunk[:, 1] * (vol * gR)

                # If fade-out completes, mark stopped (stream stays alive)
                if self._fade_state == "out" and fade <= 0.0001:
                    self._is_playing = False
                    self._is_paused = False
                    self._fade_state = "none"
                    self.status = "stopped"
                    outdata[:] = 0
                    return

        except Exception as e:
            # Never raise from callback
            with self._lock:
                self.last_error = f"Callback error: {e}"
                self.status = "error"
                self._is_playing = False
                self._is_paused = False
                self._fade_state = "none"
            outdata[:] = 0


_ENGINE: Optional[AudioEngine] = None


def get_engine(force_new: bool = False) -> AudioEngine:
    global _ENGINE
    if force_new and _ENGINE is not None:
        try:
            _ENGINE.close()
        except Exception:
            pass
        _ENGINE = None

    if _ENGINE is None:
        # device=None lets PortAudio pick the default output (usually most stable)
        _ENGINE = AudioEngine(device=None)

    return _ENGINE
