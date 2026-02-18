import time
from audio_engine import get_engine

engine = get_engine()

engine.load_wav("/var/lib/mpd/music/rain_ocean_master.wav")
engine.set_volume(0.65)
engine.set_motion(depth=0.05, period_s=80)
engine.set_fades(fade_in_s=4, fade_out_s=10)

engine.start()
print("Playing… Ctrl+C to fade out and stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    engine.stop()
    time.sleep(12)
    print("Stopped.")
