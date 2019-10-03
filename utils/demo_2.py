from datetime import datetime as dt
import os
from queue import Queue
import shutil
import sounddevice as sd
import soundfile as sf
import tempfile
import time
from utils.detector import Detector
import argparse


SAMPLE_RATE = 16000
SPLIT_AFTER = 1.5


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help='show list of audio devices and exit')
    parser.add_argument('-c', '--channels', type=int, default=1,
                        help='number of input channels')
    parser.add_argument('-t', '--subtype', type=str,
                        help='sound file subtype (e.g. "PCM_24")')
    args = parser.parse_args()

    d1 = Detector('/tmp/hi_koov_demo_1')
    d2 = Detector('/tmp/hi_koov_demo_2')
    d3 = Detector('/tmp/hi_koov_demo_3')
    d4 = Detector('/tmp/hi_koov_demo_4')
    d5 = Detector('/tmp/hi_koov_demo_5')

    while True:
      filePath = d1.getDataFolder()
      record_test_voice(args, filePath)
      result = d.evaluate()

      print('=======================================================')

      if result == 2:
          print('Hi KOOV!')
          shutil.move(filePath, os.path.join('tmp/hi_koov_archive/hi_koov'))
      elif result == 1:
          print('Unknown')
          shutil.move(filePath, os.path.join('tmp/hi_koov_archive/unknown'))
      else:
          print('Background')
          shutil.move(filePath, os.path.join('tmp/hi_koov_archive/_background_noise_'))

      print('=======================================================')

      i += 1

      time.sleep(2)



def record_test_voice(args, filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
    setattr(args, 'filePath', filePath)

    if args.list_devices:
        print(sd.query_devices())

    queue = Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        # if status:
        #     print(status, flush=True)
        queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=args.channels, callback=callback):
        counter = 0
        start_time = time.time()
        now = dt.now()
        context = {'counter': counter, 'year': now.year, 'month': now.month,
                   'day': now.day, 'hour': now.hour, 'minute': now.minute, 'second': now.second}

        filePath = args.filePath.format(**context)

        with sf.SoundFile(filePath, mode='x', samplerate=SAMPLE_RATE,
                          channels=args.channels, subtype=args.subtype) as file:
            print("Recording to: " + repr(filePath))
            while True:
                if time.time() - start_time > SPLIT_AFTER:
                    start_time += SPLIT_AFTER
                    counter += 1
                    break
                file.write(queue.get())


if __name__ == "__main__":
    main()
