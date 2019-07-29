# coding:utf-8
# Date : 2019.07.23
# author: phecda
#
#
# ********************************

import pyaudio
import wave
import time
from collections import deque
from demo.kwsNet import KwsModel


def wavSave(buffer):
    WAVE_OUTPUT_FILENAME = '../data/' + str(time.time()) + "_output.wav"
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(list(buffer)))
    wf.close()
    return WAVE_OUTPUT_FILENAME


model = KwsModel()
label_txt = '../WORK/CRNN/CRNN_nega/training/crnn_labels.txt'
model_path = '../pb/crnn1_nega.pb'


buffer=deque(maxlen=10)

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
print("starting...")
while stream.is_active():
    if len(buffer) == 10:
        data = b''.join(list(buffer))

        # pred= model.label_wav(filename, label_txt, model_path, 1)
        pred = model.label_stream(data, label_txt, model_path, 1)
        if pred[0][0] in ["marvin"]:
            filename = wavSave(buffer)
        print(pred)


stream.stop_stream()
stream.close()
p.terminate()
