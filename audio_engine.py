import threading
import time
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


@dataclass
class MotionConfig:
    # Master volume 0.0–1.0
    volume: float = 0.65

    # Stereo drift depth (0.05 = ±5% swing)
    drift_depth: float = 0.05

    # Seconds for one full L->R->L cycle
    drift_period_s: float = 80.0

    # Fade durations (defaults)
    fade_in_s: float = 4.0
    fade_out_s: float = 15.0  # set longer (e.g., 180) for sleep sessions


class AudioEngine:
    """
    Simple streaming playback engine:
      - Loads audio into memory (WAV/anything soundfile can decode)
      - Streams continuously with wrap-around loop
      - Applies stereo drift (slow balance sweep)
      - Supports fade-in/fade-out and pause/resume
      - Supports play_for(seconds) via a cancelable timer
    """

    def __init__(self, device: Optional[int] = 1):
        # Default device index = 1 (your DAC). If that changes, pass a different device.
        self.device = device
        self.cfg = MotionConfig()

        self._audio: Optional[np.ndarray] = None  # shape (N,2), float32
        self._sr: int = 44100
        self._pos: int = 0

        self._stream: Optional[sd.OutputStream] = None

        self._lock = threading.RLock()
        self._is_playing = False
        self._is_paused = False

        # Fade state
        self._fade_state = "none"  # "none" | "in" | "out"
        self._fade_t0 = 0.0

        # Drift timebase
        self._t0 = time.monotonic()

        # play_for timer cancel token
        self._timer_token = 0

    # ---------- Public API ----------

    def load_wav(self, path: str):
        """
        Loads audio using soundfile. Despite the name, this supports any format
        soundfile/libsndfile can decode on your system.
        """
        data, sr = sf.read(path, dtype="float32", always_2d=True)

        # Normalize channel count to stereo
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]

        data = np.ascontiguousarray(data, dtype=np.float32)

        with self._lock:
            self._audio = data
            self._sr = int(sr)
            self._pos = 0

    def start(self, fade_in_seconds: Optional[float] = None):
        """
        Start (or resume) playback.
        If a stream already exists, this acts like "resume" (unpauses and ensures playing).
        """
        with self._lock:
            if self._audio is None:
                raise RuntimeError("No audio loaded. Call load_wav(path) first.")

            self._is_playing = True
            self._is_paused = False

            if fade_in_seconds is not None:
                self.cfg.fade_in_s = max(0.0, float(fade_in_seconds))

            # If stream already exists, just resume
            if self._stream is not None:
                # Reset drift baseline to avoid jumps after long pauses
                self._t0 = time.monotonic()
                return

            # Begin fade-in
            self._fade_state = "in"
            self._fade_t0 = time.monotonic()
            self._t0 = time.monotonic()

            self._stream = sd.OutputStream(
                samplerate=self._sr,
                channels=2,
                dtype="float32",
                callback=self._callback,
                device=self.device,
                blocksize=0,  # PortAudio chooses
            )
            self._stream.start()

    def stop(self, fade_out_seconds: Optional[float] = None):
        """
        Graceful stop: trigger fade-out; stream closes itself when fade reaches ~0.
        """
        with self._lock:
            if fade_out_seconds is not None:
                self.cfg.fade_out_s = max(0.0, float(fade_out_seconds))

            if self._stream is None:
                self._is_playing = False
                self._is_paused = False
                self._fade_state = "none"
                return

            self._fade_state = "out"
            self._fade_t0 = time.monotonic()

    def hard_stop(self):
        """
        Immediate stop (no fade). Use when you need a guaranteed device release.
        """
        with self._lock:
            self._is_playing = False
            self._is_paused = False
            self._fade_state = "none"
            self._pos = 0

            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def pause(self):
        """
        Soft pause: stream continues but outputs silence.
        """
        with self._lock:
            if self._stream is None:
                return
            self._is_paused = True
            self._is_playing = True  # still "playing" logically, just paused

    def resume(self):
        with self._lock:
            if self._stream is None:
                # If there is no stream, start will recreate it
                return
            self._is_paused = False
            self._is_playing = True
            self._t0 = time.monotonic()

    def close(self):
        """
        Full teardown (used by get_engine(force_new=True)).
        """
        self.cancel_timer()
        self.hard_stop()

    def cancel_timer(self):
        """
        Cancel any active play_for timer.
        """
        with self._lock:
            self._timer_token += 1

    def set_volume(self, vol_0_to_1: float):
        with self._lock:
            self.cfg.volume = float(np.clip(vol_0_to_1, 0.0, 1.0))

    def set_motion(self, depth: float, period_s: float):
        with self._lock:
            self.cfg.drift_depth = float(np.clip(depth, 0.0, 0.25))
            self.cfg.drift_period_s = max(5.0, float(period_s))

    def set_fades(self, fade_in_s: float, fade_out_s: float):
        with self._lock:
            self.cfg.fade_in_s = max(0.0, float(fade_in_s))
            self.cfg.fade_out_s = max(0.0, float(fade_out_s))

    def play_for(self, seconds: int, fade_out_seconds: Optional[float] = None):
        """
        Start playback (with fade-in) and stop after N seconds (with fade-out).
        Non-blocking (spawns a daemon thread). Cancelable via cancel_timer().
        """
        # Ensure playing
        self.start()

        with self._lock:
            token = self._timer_token + 1
            self._timer_token = token

        def _timer(local_token: int):
            time.sleep(max(0, int(seconds)))
            with self._lock:
                if local_token != self._timer_token:
                    return
            self.stop(fade_out_seconds=fade_out_seconds)

        threading.Thread(target=_timer, args=(token,), daemon=True).start()

    def get_state(self):
        with self._lock:
            return {
                "is_playing": self._is_playing,
                "is_paused": self._is_paused,
                "fade_state": self._fade_state,
                "has_stream": self._stream is not None,
                "sr": self._sr,
            }

    # ---------- Internal ----------

    def _current_fade_gain(self) -> float:
        """Return fade gain in [0,1]."""
        now = time.monotonic()

        if self._fade_state == "in":
            dur = max(0.001, self.cfg.fade_in_s)
            t = (now - self._fade_t0) / dur
            x = min(1.0, max(0.0, t))
            if x >= 1.0:
                self._fade_state = "none"
            # ease-in
            return x * x

        if self._fade_state == "out":
            dur = max(0.001, self.cfg.fade_out_s)
            t = (now - self._fade_t0) / dur
            x = min(1.0, max(0.0, t))
            g = 1.0 - x
            # ease-out
            g2 = 1.0 - (1.0 - g) * (1.0 - g)
            return max(0.0, g2)

        return 1.0

    def _callback(self, outdata, frames, time_info, status):
        # Avoid prints in callback; it can glitch audio. Keep silent unless debugging.
        with self._lock:
            if not self._is_playing or self._audio is None:
                outdata[:] = 0
                return

            if self._is_paused:
                outdata[:] = 0
                return

            audio = self._audio
            n = audio.shape[0]

            # Wrap-around loop chunk extraction
            end = self._pos + frames
            if end <= n:
                chunk = audio[self._pos:end]
                self._pos = 0 if end == n else end
            else:
                first = audio[self._pos:n]
                rest = audio[0:(end - n)]
                chunk = np.vstack((first, rest))
                self._pos = end - n

            # Drift (slow stereo motion)
            t = time.monotonic() - self._t0
            depth = self.cfg.drift_depth
            period = self.cfg.drift_period_s
            s = math.sin(2.0 * math.pi * (t / period)) if depth > 0 else 0.0

            gL = 1.0 + depth * s
            gR = 1.0 - depth * s

            fade = self._current_fade_gain()
            vol = self.cfg.volume * fade

            out = chunk.copy()
            out[:, 0] *= (vol * gL)
            out[:, 1] *= (vol * gR)
            outdata[:] = out

            # If we're fading out and hit silence, close stream
            if self._fade_state == "out" and fade <= 0.0001:
                self._is_playing = False
                self._is_paused = False
                try:
                    if self._stream is not None:
                        try:
                            self._stream.stop()
                        except Exception:
                            pass
                        try:
                            self._stream.close()
                        except Exception:
                            pass
                finally:
                    self._stream = None
                    self._fade_state = "none"
                    self._pos = 0
                    outdata[:] = 0


_ENGINE: Optional[AudioEngine] = None


def get_engine(force_new: bool = False) -> AudioEngine:
    """
    Singleton getter. If force_new=True, fully closes and recreates the engine.
    """
    global _ENGINE
    if force_new and _ENGINE is not None:
        try:
            _ENGINE.close()
        except Exception:
            pass
        _ENGINE = None

    if _ENGINE is None:
        _ENGINE = AudioEngine()

    return _ENGINE
