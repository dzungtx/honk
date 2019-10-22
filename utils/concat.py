import os
from pydub import AudioSegment
import random

IN_FOLDER = '/home/dzung/data/hi_koov/negatives/'
OUT_FOLDER = '/home/dzung/data/hi_koov/negatives_random/'

for subdir in os.listdir(IN_FOLDER):
  subdirPath = os.path.join(IN_FOLDER, subdir)
  filePaths = os.listdir(subdirPath)

  for f1 in filePaths:
    sound1 = AudioSegment.from_wav(os.path.join(subdirPath, f1))
    dst = os.path.join(OUT_FOLDER, f1)
    sound1.export(dst, format="wav")
    print(dst)

    for f2 in filePaths:
      if random.random() < 0.75:
        continue

      sound2 = AudioSegment.from_wav(os.path.join(subdirPath, f2))

      if random.random() < 0.9:
        combined_sounds = sound1 + sound2
        dst = os.path.join(OUT_FOLDER, f1[:-4] + "_" + f2)
        combined_sounds.export(dst, format="wav")
        print(dst)
      else:
        for f3 in filePaths:
          if random.random() < 0.98:
            continue
          sound3 = AudioSegment.from_wav(os.path.join(subdirPath, f3))
          combined_sounds = sound1 + sound2 + sound3
          dst = os.path.join(OUT_FOLDER, f1[:-4] + "_" + f2[:-4] + "_" + f3)
          combined_sounds.export(dst, format="wav")
          print(dst)
