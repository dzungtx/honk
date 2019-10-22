import os
import random
import shutil

OUT_FOLDER = '/home/dzung/SSD500G/playground/honk/tmp/hi_koov_data/unknown'

def main():
  for f in os.listdir(OUT_FOLDER):
    os.remove(os.path.join(OUT_FOLDER, f))

  folder = '/home/dzung/SSD500G/data/hi_koov/negatives_english'
  for subfolder in os.listdir(folder):
    files = os.listdir(os.path.join(folder, subfolder))
    random.shuffle(files)
    for f in files[:5]:
      shutil.copy(os.path.join(folder, subfolder, f), os.path.join(OUT_FOLDER, subfolder + '_' + f))

  folder = '/home/dzung/SSD500G/data/hi_koov/negatives_random'
  files = os.listdir(folder)
  random.shuffle(files)
  for f in files[:390]:
      shutil.copy(os.path.join(folder, f), OUT_FOLDER)

  folder = '/home/dzung/SSD500G/data/hi_koov/negatives_similar'
  files = os.listdir(folder)
  random.shuffle(files)
  for f in files[:60]:
      shutil.copy(os.path.join(folder, f), os.path.join(OUT_FOLDER, 'similar_' + f))

  folder = '/home/dzung/SSD500G/data/hi_koov/koov'
  files = os.listdir(folder)
  random.shuffle(files)
  for f in files[:60]:
      shutil.copy(os.path.join(folder, f),
                  os.path.join(OUT_FOLDER, 'koov_' + f))


if __name__ == "__main__":
    main()
