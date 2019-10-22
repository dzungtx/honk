import numpy as np
import pyaudio
import time
import torch
import torch.nn.functional as F

from service import HiKoovLabelService, stride

INPUT_LENGTH = 16000
SAMPLE_RATE = 16000
AMPLITUDE_THRES = 0.05


class HiKoovDetector(object):
    def __init__(self):
        self.chunk_size = 16000
        self.stride_size = 500
        self.label_service = HiKoovLabelService()
        self.keyword = 'hi_koov'
        self.min_keyword_prob = 0.6

    def process(self, wav_data):
        for data in stride(wav_data, int(2 * INPUT_LENGTH * self.stride_size / 1000), 2 * INPUT_LENGTH):
            data = np.frombuffer(data, dtype=np.int16) / 32768.
            if np.amax(data) < AMPLITUDE_THRES:
                continue
            label, prob = self.label_service.label(data)
            if label == self.keyword:
                return prob
        return -1

    def _start_listening(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=self.chunk_size)

    def _stop_listening(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def start(self):
        self._start_listening()
        buf = [self.stream.read(self.chunk_size),
               self.stream.read(self.chunk_size)]
        while True:
            prob = self.process(b''.join(buf))
            if prob >= self.min_keyword_prob:
                print("Hi KOOV! ({}%)".format(round(prob * 100)), end=' ', flush=True)
                buf = [self.stream.read(self.chunk_size),
                       self.stream.read(self.chunk_size)]
            else:
                print(".", end=' ', flush=True)
                buf[0] = buf[1]
                buf[1] = self.stream.read(self.chunk_size)


def main():
    detector = HiKoovDetector()
    detector.start()


if __name__ == "__main__":
    main()
