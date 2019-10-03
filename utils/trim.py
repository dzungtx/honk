import os
from pydub import AudioSegment
import numpy as np
import random

IN_FOLDER = '/home/dzung/data/hi_koov_wake_word/positives/'
OUT_FOLDER = '/home/dzung/data/hi_koov_wake_word/positives_trim/'
PADDING = 200

filePaths = os.listdir(IN_FOLDER)

for filePath in filePaths:
    sound = AudioSegment.from_wav(IN_FOLDER + filePath)
    data = np.array(sound.get_array_of_samples())

    length = data.shape[0]
    thres = np.amax(np.abs(data)) / 10

    start = np.argmax(data > thres)
    start = max((0, start - PADDING)) * 1000 / sound.frame_rate
    end = length - np.argmax(np.flip(data > thres))
    end = min(length - 1, end + PADDING) * 1000 / sound.frame_rate

    trimmedSound = sound[start:end]
    trimmedSound.export(OUT_FOLDER + filePath, format="wav")

    reducedLength = sound.duration_seconds - trimmedSound.duration_seconds
    print(("{}: {}%\t").format(filePath, round(reducedLength /
                                               sound.duration_seconds * 100, 2)))
