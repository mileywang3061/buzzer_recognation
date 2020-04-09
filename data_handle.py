import numpy as np


def test_read_wave(path):
    """
    Read wav format audio file
    :param path: type str
    :return: type byte
    """
    with open(path, "rb") as file:
        wav = file.read()
        wav_start = 0
        if wav[0:4] == b'RIFF':
            for i in range(4, len(wav)):
                if wav[i:i + 4] == b'data':
                    wav_start = i + 8
                    break
        return wav[wav_start:len(wav)]


def wave_bytes_to_numpy(wave_bytes):
    """
    Convert binary audio into numpy format
    :param wave_bytes: type byte
    :return: type numpy
    """
    wave_data = np.fromstring(wave_bytes, dtype=np.short)
    n_frame = len(wave_data)
    n_channels = 1
    wave_data_numpy_normalized = wave_data * 1.0 / max(abs(wave_data))
    # 将音频信号规整乘每行一路通道信号的格式，即该矩阵一行为一个通道的采样点，共n_channels行
    wave_data_numpy = np.reshape(wave_data_numpy_normalized, [n_frame, n_channels]).T
    return wave_data_numpy[0]
