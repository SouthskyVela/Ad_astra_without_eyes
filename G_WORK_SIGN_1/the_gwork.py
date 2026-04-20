import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

def change_pitch(input_file, output_file, pitch_change=1.0):
    # 读取原始音频文件
    fs, data = wavfile.read(input_file)
    
    # 计算新的采样率
    new_fs = int(fs * pitch_change)
    
    # 重新采样以改变音调
    # 为了保持音频长度不变，我们需要计算新旧采样率的比例
    resample_factor = new_fs / fs
    # 重新采样
    new_data = resample(data, int(len(data) * resample_factor))
    
    # 写入新的音频文件
    wavfile.write(output_file, new_fs, new_data)

# 使用示例
input_file = "out_audio.wav"  # 输入文件路径
output_file = "gwork_1.wav"  # 输出文件路径
pitch_change = 0.25  # 改变音调的倍数，1.5 表示升高音调，0.5 表示降低音调

change_pitch(input_file, output_file, pitch_change)
