from PIL import Image
import numpy as np
import scipy.io.wavfile as wavfile

def imagetoaudio(imagepath, audiopath, duration=15, samplerate=44100):
    # 加载图像并转换为灰度
    img = Image.open(imagepath).convert('L')
    imgdata = np.array(img)

    # 计算总样本数
    total_samples = duration * samplerate

    # 创建时间数组
    t = np.linspace(0, duration, total_samples)

    # 初始化音频数组
    audio = np.zeros(total_samples)

    # 计算每列对应的样本数
    samples_per_column = total_samples // imgdata.shape[1]

    for col in range(imgdata.shape[1]):
        # 计算每列的平均亮度
        brightness = imgdata[:, col].mean()
        
        # 将亮度映射到频率
        frequency = mapbrightnesstofrequency(brightness)
        
        # 生成对应列的音频信号
        column_audio = np.sin(2 * np.pi * frequency * t[:samples_per_column])
        
        # 将生成的音频信号填充到音频数组中
        start_index = col * samples_per_column
        audio[start_index:start_index + samples_per_column] = column_audio

    # 归一化音频
    audio /= np.max(np.abs(audio))

    # 保存音频文件
    wavfile.write(audiopath, samplerate, audio.astype(np.float32))

def mapbrightnesstofrequency(brightness):
    # 将亮度值映射到频率，这里需要根据实际情况调整
    return brightness / 255.0 * 4000 + 200  # 示例映射函数，将亮度映射到200Hz到4200Hz

# 使用示例
imagepath = 'M87中心黑洞.png'
audiopath = 'M87中心黑洞.wav'
imagetoaudio(imagepath, audiopath)
