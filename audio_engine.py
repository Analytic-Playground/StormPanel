# audio_engine.py
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
    # User-facing volume. Keep as 0.0-1.0
    volume: float = 0.65

    # Gentle stereo drift (L/R gain modulation)
    drift_depth: float = 0.05          # 0.0 disables
    drift_period_s: float = 80.0

    # Fade behavior
    fade_in_s: float = 4.0
    fade_out_s: float = 15.0


class AudioEngine:
    """
    Stable stream-based loop player built on sounddevice + soundfile.

    Design goals:
    - Callback must NEVER raise (otherwise playback may die until restart)
    - All public methods are thread-safe
    - Status is always available for UI polling
    - Pause outputs silence without destroying the stream
    - Stop can fade out; hard_stop kills stream immediately
    - Auto-recovery if stream is missing/stopped when start() is called
    """

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self.cfg = MotionConfig()

        self._audio: Optional[np.ndarray] = None  # float32, shape (n,2)
        self._sr: int = 44100
        self._pos: int = 0

        self._stream: Optional[sd.OutputStream] = None

        self._lock = threading.RLock()

        # State flags
        self._is_playing: bool = False
        self._is_paused: bool = False

        # Fading
        self._fade_state: str = "none"  # "none" | "in" | "out"
        self._fade_t0: float = 0.0
        self._t0: float = time.monotonic()

        # UI/reporting
        self.current_track: Optional[str] = None
        self.status: str = "stopped"    # "playing" | "paused" | "stopped" | "error"
        self.last_error: Optional[str] = None

    # ------------------------
    # Public API
    # ------------------------

    def load_wav(self, path: str):
        """
        Load audio from disk. Accepts wav/flac/etc supported by soundfile.
        Ensures float32 stereo contiguous array.
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
            # If we were in error state, loading should clear that.
            if self.status == "error":
                self.status = "stopped"
            self.last_error = None

    def set_volume(self, volume: float):
        """
        Accepts 0.0-1.0 or 0-100; clamps safely.
        """
        v = float(volume)
        if v > 1.0:
            v = v / 100.0
        v = max(0.0, min(1.0, v))
        with self._lock:
            self.cfg.volume = v

    def start(self, fade_in_seconds: Optional[float] = None):
        """
        Start playback. Creates a stream if needed.
        """
        with self._lock:
            if self._audio is None:
                raise RuntimeError("No audio loaded.")

            self._is_playing = True
            self._is_paused = False
            self.status = "playing"
            self.last_error = None

            if fade_in_seconds is not None:
                self.cfg.fade_in_s = max(0.0, float(fade_in_seconds))

            self._fade_state = "in" if self.cfg.fade_in_s > 0 else "none"
            self._fade_t0 = time.monotonic()
            self._t0 = time.monotonic()
            self._pos = 0

            stream = self._stream

        # Start or recreate stream OUTSIDE the lock
        if stream is None or not getattr(stream, "active", False):
            self._ensure_stream()

    def stop(self, fade_out_seconds: Optional[float] = None):
        """
        Fade out, then stop/close stream once fade hits ~0.
        """
        with self._lock:
            if fade_out_seconds is not None:
                self.cfg.fade_out_s = max(0.0, float(fade_out_seconds))

            if self._stream is None:
                self._is_playing = False
                self._is_paused = False
                self._fade_state = "none"
                self.status = "stopped"
                return

            self._fade_state = "out"
            self._fade_t0 = time.monotonic()
            # Remain "playing" during fade-out for UI consistency.
            self.status = "playing" if not self._is_paused else "paused"

    def hard_stop(self):
        """
        Immediate stop: kill stream and reset position.
        """
        # Grab stream reference, clear state under lock
        with self._lock:
            self._is_playing = False
            self._is_paused = False
            self._fade_state = "none"
            self._pos = 0
            self.status = "stopped"
            stream = self._stream
            self._stream = None

        # Stop/close OUTSIDE lock
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def close(self):
        self.hard_stop()

    def pause(self):
        with self._lock:
            self._is_paused = True
            if self._is_playing:
                self.status = "paused"

    def resume(self):
        with self._lock:
            self._is_paused = False
            if self._is_playing:
                self.status = "playing"

    def toggle_pause(self):
        with self._lock:
            if not self._is_playing:
                # If nothing is playing, treat toggle as "resume/start" no-op.
                return
            if self._is_paused:
                self._is_paused = False
                self.status = "playing"
            else:
                self._is_paused = True
                self.status = "paused"

    def get_state(self) -> Dict[str, Any]:
        """
        UI polling snapshot. Keep keys stable.
        """
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
        Create and start a stream if missing. Safe to call repeatedly.
        """
        with self._lock:
            sr = self._sr
            device = self.device

        try:
            stream = sd.OutputStream(
                samplerate=sr,
                channels=2,
                dtype="float32",
                callback=self._callback,
                device=device,
                blocksize=0,
            )
            stream.start()
        except Exception as e:
            # Mark error and stop playback cleanly
            with self._lock:
                self.last_error = f"Stream start failed: {e}"
                self.status = "error"
                self._is_playing = False
                self._is_paused = False
                self._fade_state = "none"
                self._stream = None
            raise

        with self._lock:
            self._stream = stream

    def _current_fade_gain(self) -> float:
        now = time.monotonic()

        if self._fade_state == "in":
            dur = max(0.001, self.cfg.fade_in_s)
            x = min(1.0, (now - self._fade_t0) / dur)
            if x >= 1.0:
                self._fade_state = "none"
            # cosine ease in
            return 0.5 - 0.5 * math.cos(math.pi * x)

        if self._fade_state == "out":
            dur = max(0.001, self.cfg.fade_out_s)
            x = min(1.0, (now - self._fade_t0) / dur)
            g = 1.0 - x
            return max(0.0, g)

        return 1.0

    def _callback(self, outdata, frames, time_info, status):
        """
        IMPORTANT: must never raise.
        """
        try:
            with self._lock:
                # If not playing / paused / no audio, output silence
                if (not self._is_playing) or self._is_paused or (self._audio is None):
                    outdata[:] = 0
                    return

                audio = self._audio
                n = audio.shape[0]
                if n == 0:
                    outdata[:] = 0
                    return

                # Slice with wraparound
                end = self._pos + frames
                if end <= n:
                    chunk = audio[self._pos:end]
                    self._pos = 0 if end == n else end
                else:
                    first = audio[self._pos:n]
                    rest = audio[0:(end - n)]
                    chunk = np.vstack((first, rest))
                    self._pos = end - n

                # Stereo drift modulation
                t = time.monotonic() - self._t0
                depth = self.cfg.drift_depth
                period = self.cfg.drift_period_s
                s = math.sin(2.0 * math.pi * (t / period)) if depth > 0 else 0.0
                gL = 1.0 + depth * s
                gR = 1.0 - depth * s

                fade = self._current_fade_gain()
                vol = self.cfg.volume * fade

                out = chunk  # already float32
                # Apply gain without creating huge temporaries
                outdata[:, 0] = out[:, 0] * (vol * gL)
                outdata[:, 1] = out[:, 1] * (vol * gR)

                # If fade-out completed, stop stream cleanly
                if self._fade_state == "out" and fade <= 0.0001:
                    # We can't close the stream inside callback reliably.
                    # Mark state and zero output; main thread can hard_stop if needed.
                    self._is_playing = False
                    self._is_paused = False
                    self._fade_state = "none"
                    self.status = "stopped"

                    # Detach stream reference so next start() recreates it.
                    # We do NOT call stream.stop/close here.
                    self._stream = None
                    outdata[:] = 0

        except Exception as e:
            # Never raise from callback
            with self._lock:
                self.last_error = f"Callback error: {e}"
                self.status = "error"
                self._is_playing = False
                self._is_paused = False
                self._fade_state = "none"
                # Detach stream so start() recreates it
                self._stream = None
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
        # device=None lets PortAudio pick default output, usually safest
        _ENGINE = AudioEngine(device=None)

    return _ENGINE
