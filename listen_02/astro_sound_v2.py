"""
天文图像RGBL四通道音频化程序 v2.0
Astronomical Image RGBL 4-Channel Sonification v2.0

基于G_WORK_SIGN_1思想，将RGBL四个通道分别映射到四个分离的音频域
Based on G_WORK_SIGN_1, mapping RGBL channels to 4 separate audio frequency bands

作者: AI Assistant
日期: 2026-04-20
"""

import numpy as np
from astropy.io import fits
import scipy.io.wavfile as wavfile
from datetime import datetime
import os

class AstroSoundV2:
    """
    RGBL四通道音频化处理器
    
    音频域分配策略（基于人耳敏感度研究）：
    - L (Luminance/明度): 200-1200 Hz  (低频基础，承载主要结构)
    - R (Red/红色): 1200-2800 Hz  (中低频，人耳敏感区)
    - G (Green/绿色): 2800-4800 Hz  (中高频，人耳最敏感区)
    - B (Blue/蓝色): 4800-8000 Hz  (高频细节)
    """
    
    def __init__(self, duration=15, samplerate=44100):
        self.duration = duration
        self.samplerate = samplerate
        self.total_samples = duration * samplerate
        
        # 四个通道的音频域分配（基于人耳频率响应曲线优化）
        self.channel_freq_ranges = {
            'L': (200, 1200),    # 明度通道 - 低频基础
            'R': (1200, 2800),   # 红色通道 - 中低频
            'G': (2800, 4800),   # 绿色通道 - 中高频（人耳最敏感）
            'B': (4800, 8000),   # 蓝色通道 - 高频细节
        }
        
        print(f"[初始化] AstroSoundV2 音频处理器")
        print(f"  采样率: {samplerate} Hz")
        print(f"  音频时长: {duration} 秒")
        print(f"  总样本数: {self.total_samples}")
        print(f"\n[音频域分配]")
        for ch, (fmin, fmax) in self.channel_freq_ranges.items():
            print(f"  {ch}通道: {fmin}-{fmax} Hz")
    
    def load_fits(self, filepath):
        """加载FITS文件并返回数据"""
        try:
            with fits.open(filepath) as hdul:
                data = hdul[0].data
                # 处理可能的维度问题
                if data.ndim == 3:
                    data = data[0] if data.shape[0] == 1 else np.mean(data, axis=0)
                return data.astype(np.float32)
        except Exception as e:
            print(f"[错误] 无法加载FITS文件 {filepath}: {e}")
            return None
    
    def normalize_data(self, data):
        """将数据归一化到0-1范围"""
        min_val = np.percentile(data, 1)  # 使用1%分位数避免极端值
        max_val = np.percentile(data, 99)  # 使用99%分位数
        normalized = (data - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    
    def map_value_to_frequency(self, value, freq_range):
        """将归一化值(0-1)映射到指定频率范围"""
        fmin, fmax = freq_range
        return fmin + value * (fmax - fmin)
    
    def column_to_audio(self, column_data, freq_range, samples_per_column):
        """将单列数据转换为音频信号"""
        # 计算列的平均亮度
        brightness = np.mean(column_data)
        
        # 映射到频率
        frequency = self.map_value_to_frequency(brightness, freq_range)
        
        # 生成正弦波（添加轻微包络避免爆音）
        t = np.linspace(0, samples_per_column / self.samplerate, samples_per_column)
        audio = np.sin(2 * np.pi * frequency * t)
        
        # 添加淡入淡出
        fade_samples = min(100, samples_per_column // 10)
        if fade_samples > 0:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return audio, frequency
    
    def process_channel(self, data, channel_name, freq_range):
        """处理单个通道，生成音频"""
        print(f"\n[处理 {channel_name} 通道]")
        print(f"  原始数据范围: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  数据形状: {data.shape}")
        
        # 数据归一化
        normalized = self.normalize_data(data)
        print(f"  归一化后范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
        
        # 计算每列样本数
        samples_per_column = self.total_samples // data.shape[1]
        print(f"  每列样本数: {samples_per_column}")
        
        # 初始化音频数组
        audio = np.zeros(self.total_samples)
        freq_history = []
        
        # 按列扫描
        for col in range(data.shape[1]):
            column_data = normalized[:, col]
            column_audio, freq = self.column_to_audio(
                column_data, freq_range, samples_per_column
            )
            freq_history.append(freq)
            
            # 填充到音频数组
            start_idx = col * samples_per_column
            end_idx = min(start_idx + samples_per_column, self.total_samples)
            audio[start_idx:end_idx] = column_audio[:end_idx - start_idx]
        
        # 归一化
        max_amp = np.max(np.abs(audio))
        if max_amp > 0:
            audio /= max_amp
        
        print(f"  频率范围: {min(freq_history):.1f} - {max(freq_history):.1f} Hz")
        print(f"  平均频率: {np.mean(freq_history):.1f} Hz")
        
        return audio.astype(np.float32), freq_history
    
    def mix_channels(self, audio_dict, weights=None):
        """混合多个通道的音频"""
        if weights is None:
            weights = {ch: 0.25 for ch in audio_dict.keys()}
        
        mixed = np.zeros(self.total_samples)
        for ch, audio in audio_dict.items():
            mixed += audio * weights.get(ch, 0.25)
        
        # 归一化
        max_amp = np.max(np.abs(mixed))
        if max_amp > 0:
            mixed /= max_amp
        
        return mixed.astype(np.float32)
    
    def process_rgbL(self, data_dir, output_dir):
        """处理RGBL四个通道并生成音频"""
        
        channels = ['R', 'G', 'B', 'L']
        audio_results = {}
        freq_histories = {}
        
        print("=" * 60)
        print("天文图像RGBL四通道音频化 v2.0")
        print("=" * 60)
        
        # 加载并处理每个通道
        for ch in channels:
            fits_path = os.path.join(data_dir, f"{ch}.fits")
            
            if not os.path.exists(fits_path):
                print(f"[警告] 找不到文件: {fits_path}")
                continue
            
            # 加载数据
            data = self.load_fits(fits_path)
            if data is None:
                continue
            
            # 处理通道
            freq_range = self.channel_freq_ranges[ch]
            audio, freq_hist = self.process_channel(data, ch, freq_range)
            
            audio_results[ch] = audio
            freq_histories[ch] = freq_hist
            
            # 保存单个通道音频
            output_path = os.path.join(output_dir, f"M38_{ch}_channel.wav")
            wavfile.write(output_path, self.samplerate, audio)
            print(f"  已保存: {output_path}")
        
        # 合成所有通道
        if len(audio_results) > 0:
            print("\n[合成音频]")
            mixed_audio = self.mix_channels(audio_results)
            
            mixed_path = os.path.join(output_dir, "M38_RGBL_mixed.wav")
            wavfile.write(mixed_path, self.samplerate, mixed_audio)
            print(f"  已保存合成音频: {mixed_path}")
            
            audio_results['mixed'] = mixed_audio
        
        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        
        return audio_results, freq_histories


def main():
    """主程序"""
    # 路径设置
    data_dir = r"F:\my_github\listen_to_the_universe\TestData\RGBL\M38"
    output_dir = r"F:\my_github\listen_to_the_universe\listen_02"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建处理器并运行
    processor = AstroSoundV2(duration=20, samplerate=44100)
    audio_results, freq_histories = processor.process_rgbL(data_dir, output_dir)
    
    return audio_results, freq_histories


if __name__ == "__main__":
    audio_results, freq_histories = main()
