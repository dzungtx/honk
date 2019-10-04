from datetime import datetime as dt
import os
from queue import Queue
import shutil
import sounddevice as sd
import soundfile as sf
import tempfile
import time
from utils.detector import Detector


SAMPLE_RATE = 16000
SPLIT_AFTER = 1.5
DEMO_FOLDER = './tmp/hi_koov_demo/hi_koov'
AUDIO_SAVE_PATH = 'tmp/hi_koov_demo/hi_koov/voice.wav'


def main():
    for fileName in os.listdir(DEMO_FOLDER):
        os.unlink(os.path.join(DEMO_FOLDER, fileName))

    # for folder in ['tmp/hi_koov_archive/_background_noise_', 'tmp/hi_koov_archive/unknown', 'tmp/hi_koov_archive/hi_koov', 'tmp/hi_koov_demo/hi_koov']:
    #     for the_file in os.listdir(folder):
    #         file_path = os.path.join(folder, the_file)
    #         try:
    #             if os.path.isfile(file_path):
    #                 os.unlink(file_path)
    #             # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    #         except Exception as e:
    #             print(e)

    d = Detector()

    # i = 1

    while True:
        # filePath = os.path.join(
        #     'tmp/hi_koov_demo/hi_koov/', 'voice-{}.wav'.format(i))
        # record_test_voice(filePath)
        record_test_voice(AUDIO_SAVE_PATH)

        t1 = time.time()

        result = d.evaluate()

        t2 = time.time()

        print('=======================================================')

        if result == 2:
            print('Hi KOOV!\t- Inference time: {}s'.format(round(t2 - t1, 2)))
        else:
            print('Unknown\t- Inference time: {}s'.format(round(t2 - t1, 2)))

        # if result == 2:
        #     print('Hi KOOV!')
        #     shutil.move(filePath, os.path.join('tmp/hi_koov_archive/hi_koov'))
        # elif result == 1:
        #     print('Unknown')
        #     shutil.move(filePath, os.path.join('tmp/hi_koov_archive/unknown'))
        # else:
        #     print('Background')
        #     shutil.move(filePath, os.path.join(
        #         'tmp/hi_koov_archive/_background_noise_'))

        print('=======================================================')

        # i += 1

        time.sleep(2)


def record_test_voice(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)

    queue = Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        # if status:
        #     print(status, flush=True)
        queue.put(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        counter = 0
        startTime = time.time()
        now = dt.now()

        context = {'counter': counter, 'year': now.year, 'month': now.month,
                   'day': now.day, 'hour': now.hour, 'minute': now.minute, 'second': now.second}
        filePath = filePath.format(**context)

        with sf.SoundFile(filePath, mode='x', samplerate=SAMPLE_RATE, channels=1) as file:
            print("Start recording ...")
            while True:
                if time.time() - startTime > SPLIT_AFTER:
                    startTime += SPLIT_AFTER
                    counter += 1
                    break
                file.write(queue.get())
            print("Stop recording")


if __name__ == "__main__":
    main()
