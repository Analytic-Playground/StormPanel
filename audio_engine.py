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
    volume: float = 0.65
    drift_depth: float = 0.05
    drift_period_s: float = 80.0
    fade_in_s: float = 4.0
    fade_out_s: float = 15.0


class AudioEngine:
    def __init__(self, device: Optional[int] = 1):
        self.device = device
        self.cfg = MotionConfig()

        self._audio: Optional[np.ndarray] = None
        self._sr: int = 44100
        self._pos: int = 0

        self._stream: Optional[sd.OutputStream] = None

        self._lock = threading.RLock()
        self._is_playing = False
        self._is_paused = False

        self._fade_state = "none"
        self._fade_t0 = 0.0
        self._t0 = time.monotonic()

        self._timer_token = 0

    # ------------------------
    # Public API
    # ------------------------

    def load_wav(self, path: str):
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

    def start(self, fade_in_seconds: Optional[float] = None):
        with self._lock:
            if self._audio is None:
                raise RuntimeError("No audio loaded.")

            self._is_playing = True
            self._is_paused = False

            if fade_in_seconds is not None:
                self.cfg.fade_in_s = max(0.0, float(fade_in_seconds))

            self._fade_state = "in" if self.cfg.fade_in_s > 0 else "none"
            self._fade_t0 = time.monotonic()
            self._t0 = time.monotonic()

            if self._stream is not None:
                self._pos = 0
                return

            sr = self._sr
            device = self.device

        stream = sd.OutputStream(
            samplerate=sr,
            channels=2,
            dtype="float32",
            callback=self._callback,
            device=device,
            blocksize=0,
        )
        stream.start()

        with self._lock:
            self._stream = stream
            self._pos = 0

    def stop(self, fade_out_seconds: Optional[float] = None):
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

    def close(self):
        self.hard_stop()

    # ------------------------
    # Internal
    # ------------------------

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
        with self._lock:
            if not self._is_playing or self._audio is None or self._is_paused:
                outdata[:] = 0
                return

            audio = self._audio
            n = audio.shape[0]

            end = self._pos + frames
            if end <= n:
                chunk = audio[self._pos:end]
                self._pos = 0 if end == n else end
            else:
                first = audio[self._pos:n]
                rest = audio[0:(end - n)]
                chunk = np.vstack((first, rest))
                self._pos = end - n

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

            if self._fade_state == "out" and fade <= 0.0001:
                self.hard_stop()
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
        _ENGINE = AudioEngine()

    return _ENGINE
