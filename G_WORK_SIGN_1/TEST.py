from PIL import Image
import numpy as np
import scipy.io.wavfile as wavfile

def image_to_audio(image_path, audio_path, sample_rate=44100):
    # 打开图像并转换为灰度
    img = Image.open(image_path).convert('L')
    img_data = np.array(img)

    # 创建时间数组
    t = np.linspace(0, 100, sample_rate)  # 1秒的音频

    # 初始化音频数组
    audio = np.zeros(sample_rate)

    # 将每列转换为音频频率
    for col in range(img_data.shape[1]):
        brightness = img_data[:, col].mean()  # 计算每列的平均亮度
        frequency = map_brightness_to_frequency(brightness)  # 将亮度映射到频率
        audio += np.sin(2 * np.pi * frequency * t)

    # 归一化音频
    audio /= np.max(np.abs(audio))

    # 保存音频文件
    wavfile.write(audio_path, sample_rate, audio)

def map_brightness_to_frequency(brightness):
    # 将亮度值映射到频率，这里需要根据实际情况调整
    return brightness * 20  # 示例映射函数

# 使用示例
image_path = '屏幕截图 2025-05-14 191930.jpg'
audio_path = 'audio_1.wav'
image_to_audio(image_path, audio_path)
