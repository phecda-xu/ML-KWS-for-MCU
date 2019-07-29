import pyaudio
import wave
import time
from collections import deque

def wavSave(buffer):
    WAVE_OUTPUT_FILENAME = str(time.time()) + "_output.wav"
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(list(buffer)))
    wf.close()


buffer=deque(maxlen=20)

CHUNK = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, status):
    buffer.append(in_data)
    return (in_data, pyaudio.paContinue)


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

stream.start_stream()
while stream.is_active():
    print("starting...")
    print(len(buffer))
    if len(buffer)> 5:
        # data = b''.join(list(buffer))
        wavSave(buffer)
        print("saved!")
        time.sleep(5)

stream.stop_stream()
stream.close()
p.terminate()

