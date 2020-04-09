import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
import time
import random
import sys
import io
import cv2
from PIL import Image
import string

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from buzzer_algorithm.detect_algo.data_handle import wave_bytes_to_numpy, test_read_wave
from buzzer_algorithm.detect_algo.buzzer_algo_process import get_information, rules

# the time period of buzzer detection
UNIT_PREDICT_TIME = 15
# the time period of user giving
WAVE_TIME_GIVING = 15
# the sampling rate of the audio
SAMPLE_RATE = 16000
# the byte size of the 15s audio data
# PREDICT_BYTES_DATA_LENGTH = int((SAMPLE_RATE * WAVE_DATA_PREDICT_TIME))
UNIT_PREDICT_NUMPY_DATA_LENGTH = int((SAMPLE_RATE * WAVE_TIME_GIVING))
# intermediate variable used to hold audio data
# g_middle_temp_wave_data = b""
g_middle_numpy_data = None

# buzzer detection status code
NO_BUZZER_SOUND_FLAG = 0
BUZZER_SOUND_FLAG = 1
AUDIO_NO_ENOUGH_FLAG = 2

# process pool parameter
MAX_PROCESS_NUMBERS = 4
g_process_pools = None

# configuration dict of buzzer detection in different scenes
g_multi_scene_time_period_dict = None

total_time = list()


def init_buzzer_params():
    global MAX_PROCESS_NUMBERS
    global g_multi_scene_time_period_dict
    global g_process_pools
    global g_middle_numpy_data

    g_middle_numpy_data = np.zeros((0), dtype=np.short)
    g_process_pools = ProcessPoolExecutor(MAX_PROCESS_NUMBERS)
    g_multi_scene_time_period_dict = {"15s_detect_period": 1, }


def buzzer_predict_online(wave_bytes, buzzer_type="15s_detect_period"):
    """
    real-time detection of buzzer sound
    :param wave_bytes: type byte
    :param buzzer_type: type str
    :return: type int
    """

    # start_time = time.time()
    global SAMPLE_RATE
    global g_middle_numpy_data
    global AUDIO_NO_ENOUGH_FLAG
    global UNIT_PREDICT_NUMPY_DATA_LENGTH
    global g_multi_scene_time_period_dict
    global UNIT_PREDICT_TIME
    if len(wave_bytes) != 0:
        wave_data_numpy = wave_bytes_to_numpy(wave_bytes)
        g_middle_numpy_data = np.concatenate((g_middle_numpy_data, wave_data_numpy))
    g_middle_numpy_data_length = len(g_middle_numpy_data)
    multiple_time = g_multi_scene_time_period_dict.get(buzzer_type)
    predict_numpy_data_length = multiple_time * UNIT_PREDICT_NUMPY_DATA_LENGTH
    if g_middle_numpy_data_length < predict_numpy_data_length:
        return AUDIO_NO_ENOUGH_FLAG
    elif g_middle_numpy_data_length == predict_numpy_data_length:
        data_numpy = g_middle_numpy_data
        g_middle_numpy_data = np.zeros((0), dtype=np.short)
    else:
        data_numpy = g_middle_numpy_data[0:g_middle_numpy_data_length]
        g_middle_numpy_data = g_middle_numpy_data[predict_numpy_data_length:g_middle_numpy_data_length]
    NFFT, frame_size, overlap_size = get_information(SAMPLE_RATE)

    # start_time_1 = time.time()
    # print("handle data waste time: {}".format(start_time_1 - start_time))
    imag = draw_buzzer_gca(data_numpy, NFFT, SAMPLE_RATE, frame_size, overlap_size, multiple_time)
    # start_time_2 = time.time()
    # print("draw_buzzer_gca waste time: {}".format(start_time_2 - start_time_1))
    buzzer_detect_flag = find_buzzer_position(imag, multiple_time)
    # start_time_3 = time.time()
    # print("buzzer_detect_flag waste time: {}".format(start_time_3 - start_time_2))
    # end_time_1 = time.time()
    # total_time.append(end_time_1 - start_time)

    return buzzer_detect_flag


def buzzer_predict_finish():
    """global variables are released at the end of the session"""
    global g_middle_numpy_data
    g_middle_numpy_data = np.zeros((0), dtype=np.short)

    print(np.mean(total_time))


def draw_buzzer_gca(wave_data_numpy, NFFT, sample_rate, frame_size, overlap_size, multiple_time):
    plt.rcParams['figure.figsize'] = (12.8 * multiple_time, 9.6)
    plt.rcParams['savefig.dpi'] = 100 * 1
    plt.specgram(wave_data_numpy, NFFT=NFFT, Fs=sample_rate, window=np.hanning(M=frame_size), noverlap=overlap_size,
                 mode='default', scale_by_freq=True, sides='default', scale='dB', xextent=None)  # 绘制频谱图
    fig = plt.gcf()
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # 图片保存的位置
    # start_time = time.time()
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    plt.close()
    im = Image.open(buffer).convert("RGB")
    image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    buffer.close()
    # print("-----------", time.time()-start_time)
    return image


def find_buzzer_position(image_data_numpy, multiple_time):
    """
    :param image_data_numpy: type numpy
    :param multiple_time: type float
    :return: type int
    """
    futures_list = []
    for t in range(165, 190):
        future = g_process_pools.submit(rules, image_data_numpy, t, multiple_time)
        futures_list.append(future)
    for future in futures_list:
        temp = future.result()
        if temp is True:
            return BUZZER_SOUND_FLAG
    return NO_BUZZER_SOUND_FLAG


if __name__ == '__main__':
    init_buzzer_params()
    waves_dirname = os.path.join(BASE_DIR, "buzzer_algorithm", "test_wave")
    list_file = os.listdir(waves_dirname)
    wave_data = test_read_wave(os.path.join(waves_dirname, list_file[0]))
    for index in range(0, len(wave_data), 1600 * 5 * 2):
        print(buzzer_predict_online(wave_data[index: index + 1600 * 5 * 2]))
    print(buzzer_predict_finish())
    # print(np.mean(total_time))
