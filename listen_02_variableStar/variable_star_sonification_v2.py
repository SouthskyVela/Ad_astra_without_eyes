"""
造父变星光变曲线音频化程序 v2.0
Cepheid Variable Star Light Curve Sonification v2.0

修复问题:
1. scheme2频域音频无声音 - 修复频率映射
2. 提高曲线拟合精度
3. 添加原始数据与拟合数据对比图

Author: AI Assistant
Date: 2026-04-21
"""

import numpy as np
import pandas as pd
from scipy import interpolate, fft
from scipy.signal import spectrogram, find_peaks
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import os
from datetime import datetime

class VariableStarSonificationV2:
    """
    造父变星光变曲线音频化处理器 v2.0
    """
    
    def __init__(self, duration=30, samplerate=44100):
        self.duration = duration
        self.samplerate = samplerate
        self.total_samples = int(duration * samplerate)
        
        # 频率映射范围
        self.freq_min = 200   # Hz
        self.freq_max = 2000  # Hz
        
        print("=" * 60)
        print("造父变星光变曲线音频化处理器 v2.0")
        print("=" * 60)
        print(f"音频时长: {duration} 秒")
        print(f"采样率: {samplerate} Hz")
        print(f"频率映射范围: {self.freq_min}-{self.freq_max} Hz")
        print("=" * 60)
    
    def load_data(self, filepath):
        """加载AAVSO格式的光变曲线数据"""
        print(f"\n[数据加载] {filepath}")
        
        df = pd.read_csv(filepath, sep='\t', skipinitialspace=True)
        jd = df['JD'].values
        mag = df['Magnitude'].values
        
        # 排序
        sort_idx = np.argsort(jd)
        jd, mag = jd[sort_idx], mag[sort_idx]
        
        # 去除重复时间点
        unique_jd, unique_mag = [], []
        for i in range(len(jd)):
            if i == 0 or not np.isclose(jd[i], jd[i-1], rtol=1e-10):
                unique_jd.append(jd[i])
                unique_mag.append(mag[i])
            else:
                unique_mag[-1] = (unique_mag[-1] + mag[i]) / 2
        
        jd = np.array(unique_jd)
        mag = np.array(unique_mag)
        time = jd - jd[0]
        
        print(f"  观测点数: {len(time)}")
        print(f"  时间跨度: {time[-1] - time[0]:.2f} 天")
        print(f"  星等范围: [{mag.min():.3f}, {mag.max():.3f}]")
        
        return time, mag
    
    def magnitude_to_frequency(self, magnitude):
        """星等映射到频率"""
        mag_min, mag_max = magnitude.min(), magnitude.max()
        normalized = (magnitude - mag_min) / (mag_max - mag_min)
        frequency = self.freq_max - normalized * (self.freq_max - self.freq_min)
        return frequency
    
    # ========================================================================
    # 方案1: 离散处理
    # ========================================================================
    
    def scheme1_discrete_time_domain(self, time, mag, output_path):
        """方案1: 时域音频化（离散数据）"""
        print("\n" + "-" * 60)
        print("[方案1] 时域音频化 - 离散数据 → 离散音高")
        print("-" * 60)
        
        frequencies = self.magnitude_to_frequency(mag)
        print(f"  频率范围: {frequencies.min():.1f} - {frequencies.max():.1f} Hz")
        
        # 计算时间间隔
        time_normalized = time / time[-1] * self.duration
        time_diffs = np.diff(time_normalized)
        time_diffs = np.append(time_diffs, time_diffs[-1])
        
        # 生成音频
        audio = np.array([])
        for i in range(len(time)):
            samples = int(time_diffs[i] * self.samplerate)
            if samples == 0:
                continue
            
            t = np.linspace(0, time_diffs[i], samples)
            segment = np.sin(2 * np.pi * frequencies[i] * t)
            
            # 淡入淡出
            fade = min(50, samples // 10)
            if fade > 0:
                segment[:fade] *= np.linspace(0, 1, fade)
                segment[-fade:] *= np.linspace(1, 0, fade)
            
            audio = np.append(audio, segment)
        
        # 调整长度
        if len(audio) > self.total_samples:
            audio = audio[:self.total_samples]
        elif len(audio) < self.total_samples:
            audio = np.pad(audio, (0, self.total_samples - len(audio)))
        
        audio /= np.max(np.abs(audio))
        wavfile.write(output_path, self.samplerate, audio.astype(np.float32))
        print(f"  已保存: {output_path}")
        
        return audio, frequencies
    
    def scheme1_discrete_frequency_domain(self, time, mag, output_path):
        """方案1: 频域音频化（DFT频谱）"""
        print("\n" + "-" * 60)
        print("[方案1] 频域音频化 - DFT频谱")
        print("-" * 60)
        
        N = len(mag)
        dft = fft.fft(mag)
        dft_magnitude = np.abs(dft[:N//2])
        dft_phase = np.angle(dft[:N//2])
        freqs = fft.fftfreq(N, np.median(np.diff(time)))[:N//2]
        
        print(f"  DFT点数: {N}")
        print(f"  频谱范围: {freqs[1]:.6f} - {freqs[-1]:.6f} cycles/day")
        
        # 将频谱映射到音频（从低频到高频播放）
        # 每个频点持续固定时间
        segment_duration = self.duration / (len(dft_magnitude) - 1)
        samples_per_segment = int(segment_duration * self.samplerate)
        
        audio = np.array([])
        for i in range(1, len(dft_magnitude)):  # 跳过DC
            # 映射到音频频率范围
            progress = i / len(dft_magnitude)
            audio_freq = self.freq_min + progress * (self.freq_max - self.freq_min)
            
            t = np.linspace(0, segment_duration, samples_per_segment)
            amplitude = dft_magnitude[i] / (np.max(dft_magnitude) + 1e-10)
            segment = amplitude * np.sin(2 * np.pi * audio_freq * t + dft_phase[i])
            
            audio = np.append(audio, segment)
        
        # 调整长度
        if len(audio) > self.total_samples:
            audio = audio[:self.total_samples]
        elif len(audio) < self.total_samples:
            audio = np.pad(audio, (0, self.total_samples - len(audio)))
        
        audio /= np.max(np.abs(audio) + 1e-10)
        wavfile.write(output_path, self.samplerate, audio.astype(np.float32))
        print(f"  已保存: {output_path}")
        
        return audio, dft_magnitude, freqs
    
    # ========================================================================
    # 方案2: 连续处理（高精度）
    # ========================================================================
    
    def fourier_series_fit(self, time, mag, n_harmonics=20):
        """
        使用傅里叶级数拟合光变曲线
        
        数学原理:
        f(t) = a0/2 + Σ[an*cos(2π*n*t/T) + bn*sin(2π*n*t/T)]
        
        其中:
        - T: 光变周期
        - a0: 直流分量
        - an, bn: 傅里叶系数
        - n: 谐波次数
        """
        # 估计周期（使用DFT找主峰）
        N = len(mag)
        dft = fft.fft(mag - np.mean(mag))
        freqs = fft.fftfreq(N, np.median(np.diff(time)))
        
        # 找主峰（排除DC）
        positive_freqs = freqs[1:N//2]
        positive_dft = np.abs(dft[1:N//2])
        peak_idx = np.argmax(positive_dft)
        main_freq = positive_freqs[peak_idx]
        period = 1.0 / main_freq
        
        print(f"  估计周期: {period:.4f} 天")
        print(f"  基频: {main_freq:.6f} cycles/day")
        
        # 计算傅里叶系数
        T = time[-1] - time[0]
        omega = 2 * np.pi / period
        
        a0 = 2 * np.mean(mag)
        an = []
        bn = []
        
        for n in range(1, n_harmonics + 1):
            an_val = 2 * np.mean(mag * np.cos(n * omega * time)) / N * len(time)
            bn_val = 2 * np.mean(mag * np.sin(n * omega * time)) / N * len(time)
            an.append(an_val)
            bn.append(bn_val)
        
        return a0, an, bn, period, omega
    
    def evaluate_fourier_series(self, time_eval, a0, an, bn, omega):
        """计算傅里叶级数在给定时间点的值"""
        result = a0 / 2 * np.ones_like(time_eval)
        for n, (a, b) in enumerate(zip(an, bn), 1):
            result += a * np.cos(n * omega * time_eval) + b * np.sin(n * omega * time_eval)
        return result
    
    def scheme2_continuous_time_domain(self, time, mag, output_path):
        """方案2: 时域音频化（傅里叶级数拟合 → 连续音高）"""
        print("\n" + "-" * 60)
        print("[方案2] 时域音频化 - 傅里叶级数拟合 → 连续音高")
        print("-" * 60)
        
        # 傅里叶级数拟合
        a0, an, bn, period, omega = self.fourier_series_fit(time, mag, n_harmonics=20)
        
        # 使用傅里叶级数生成高密度数据
        N_fine = 50000
        time_fine = np.linspace(time[0], time[-1], N_fine)
        mag_fine = self.evaluate_fourier_series(time_fine, a0, an, bn, omega)
        
        print(f"  傅里叶级数谐波数: 20")
        print(f"  插值点数: {N_fine}")
        
        frequencies = self.magnitude_to_frequency(mag_fine)
        print(f"  频率范围: {frequencies.min():.1f} - {frequencies.max():.1f} Hz")
        
        # 生成Chirp信号
        time_audio = np.linspace(0, self.duration, self.total_samples)
        freq_interp = interpolate.interp1d(
            np.linspace(0, self.duration, len(frequencies)),
            frequencies,
            kind='cubic',
            fill_value='extrapolate'
        )
        freqs_at_samples = freq_interp(time_audio)
        
        # 相位积分
        phase = 2 * np.pi * np.cumsum(freqs_at_samples) / self.samplerate
        audio = np.sin(phase)
        audio /= np.max(np.abs(audio))
        
        wavfile.write(output_path, self.samplerate, audio.astype(np.float32))
        print(f"  已保存: {output_path}")
        
        return audio, frequencies, time_fine, mag_fine, (a0, an, bn, period, omega)
    
    def scheme2_continuous_frequency_domain(self, time, mag, output_path):
        """方案2: 频域音频化（基于傅里叶级数的频谱）"""
        print("\n" + "-" * 60)
        print("[方案2] 频域音频化 - 傅里叶级数频谱")
        print("-" * 60)
        
        # 傅里叶级数拟合
        a0, an, bn, period, omega = self.fourier_series_fit(time, mag, n_harmonics=50)
        
        # 使用傅里叶级数生成高密度数据
        N = 65536
        time_fine = np.linspace(time[0], time[-1], N)
        mag_fine = self.evaluate_fourier_series(time_fine, a0, an, bn, omega)
        
        print(f"  傅里叶级数谐波数: 50")
        print(f"  插值点数: {N}")
        
        # FFT
        window = np.hanning(N)
        fft_result = fft.fft(mag_fine * window)
        fft_magnitude = np.abs(fft_result[:N//2])
        fft_phase = np.angle(fft_result[:N//2])
        
        dt = (time[-1] - time[0]) / N
        freqs = fft.fftfreq(N, dt)[:N//2]
        
        print(f"  频率分辨率: {freqs[1]:.8f} cycles/day")
        print(f"  频谱范围: {freqs[1]:.6f} - {freqs[-1]:.6f} cycles/day")
        
        # 将频谱映射到音频
        max_freq_idx = min(len(fft_magnitude), 5000)
        segment_duration = self.duration / (max_freq_idx - 1)
        samples_per_segment = int(segment_duration * self.samplerate)
        
        audio = np.array([])
        for i in range(1, max_freq_idx):
            progress = i / max_freq_idx
            audio_freq = self.freq_min + progress * (self.freq_max - self.freq_min)
            
            t = np.linspace(0, segment_duration, samples_per_segment)
            amplitude = fft_magnitude[i] / (np.max(fft_magnitude) + 1e-10)
            amplitude = np.clip(amplitude * 2, 0.1, 1.0)
            
            segment = amplitude * np.sin(2 * np.pi * audio_freq * t + fft_phase[i])
            audio = np.append(audio, segment)
        
        if len(audio) > self.total_samples:
            audio = audio[:self.total_samples]
        elif len(audio) < self.total_samples:
            audio = np.pad(audio, (0, self.total_samples - len(audio)))
        
        audio /= np.max(np.abs(audio) + 1e-10)
        wavfile.write(output_path, self.samplerate, audio.astype(np.float32))
        print(f"  已保存: {output_path}")
        
        return audio, fft_magnitude, freqs
    
    # ========================================================================
    # 可视化（添加对比图）
    # ========================================================================
    
    def visualize_results(self, time, mag, output_dir):
        """生成完整可视化图表（包含原始数据与拟合对比）"""
        print("\n" + "=" * 60)
        print("[可视化] 生成分析图表")
        print("=" * 60)
        
        # 创建大图
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 原始光变曲线
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(time, mag, 'ko', markersize=4, label='Observations (N=61)')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Delta Cephei Light Curve - Raw Data')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 原始数据 vs 傅里叶级数拟合对比
        ax2 = plt.subplot(3, 2, 2)
        
        # 傅里叶级数拟合
        a0, an, bn, period, omega = self.fourier_series_fit(time, mag, n_harmonics=20)
        time_fine = np.linspace(time[0], time[-1], 10000)
        mag_fourier = self.evaluate_fourier_series(time_fine, a0, an, bn, omega)
        
        ax2.plot(time, mag, 'ko', markersize=5, alpha=0.6, label='Raw Data')
        ax2.plot(time_fine, mag_fourier, 'r-', linewidth=1.5, label=f'Fourier Series (n=20, T={period:.2f}d)')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Raw Data vs Fourier Series Fit')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 原始数据DFT频谱
        ax3 = plt.subplot(3, 2, 3)
        N_raw = len(mag)
        dft_raw = fft.fft(mag)
        dft_mag_raw = np.abs(dft_raw[:N_raw//2])
        freqs_raw = fft.fftfreq(N_raw, np.median(np.diff(time)))[:N_raw//2]
        
        ax3.semilogy(freqs_raw[1:], dft_mag_raw[1:], 'b-', linewidth=1.5)
        ax3.set_xlabel('Frequency (cycles/day)')
        ax3.set_ylabel('|DFT|')
        ax3.set_title(f'DFT of Raw Data (N={N_raw})')
        ax3.grid(True, alpha=0.3)
        
        # 标注主峰
        peaks, _ = find_peaks(dft_mag_raw[1:], height=np.max(dft_mag_raw[1:])*0.3)
        for peak in peaks[:3]:
            ax3.axvline(freqs_raw[peak+1], color='r', linestyle='--', alpha=0.5)
            ax3.annotate(f'{freqs_raw[peak+1]:.3f}', 
                        xy=(freqs_raw[peak+1], dft_mag_raw[peak+1]),
                        fontsize=8, color='red')
        
        # 4. 傅里叶级数拟合后的FFT频谱
        ax4 = plt.subplot(3, 2, 4)
        N_fit = 16384
        time_fit = np.linspace(time[0], time[-1], N_fit)
        mag_fourier_fit = self.evaluate_fourier_series(time_fit, a0, an, bn, omega)
        window = np.hanning(N_fit)
        fft_fit = fft.fft(mag_fourier_fit * window)
        fft_mag_fit = np.abs(fft_fit[:N_fit//2])
        freqs_fit = fft.fftfreq(N_fit, (time[-1]-time[0])/N_fit)[:N_fit//2]
        
        ax4.semilogy(freqs_fit[1:], fft_mag_fit[1:], 'm-', linewidth=1)
        ax4.set_xlabel('Frequency (cycles/day)')
        ax4.set_ylabel('|FFT|')
        ax4.set_title(f'FFT of Fourier Fitted Data (N={N_fit})')
        ax4.grid(True, alpha=0.3)
        
        # 5. 频谱对比（叠加）
        ax5 = plt.subplot(3, 2, 5)
        ax5.semilogy(freqs_raw[1:], dft_mag_raw[1:]/np.max(dft_mag_raw), 
                     'b-', linewidth=2, label='Raw DFT (normalized)')
        ax5.semilogy(freqs_fit[1:len(freqs_fit)//10], 
                     fft_mag_fit[1:len(freqs_fit)//10]/np.max(fft_mag_fit), 
                     'r-', linewidth=1, alpha=0.7, label='Fitted FFT (normalized)')
        ax5.set_xlabel('Frequency (cycles/day)')
        ax5.set_ylabel('Normalized Magnitude')
        ax5.set_title('Spectrum Comparison: Raw vs Fitted')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.set_xlim(0, freqs_raw[-1])
        
        # 6. 残差分析（傅里叶级数拟合）
        ax6 = plt.subplot(3, 2, 6)
        mag_fourier_at_obs = self.evaluate_fourier_series(time, a0, an, bn, omega)
        residual = mag - mag_fourier_at_obs
        ax6.plot(time, residual, 'g.', markersize=4, alpha=0.6)
        ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Time (days)')
        ax6.set_ylabel('Residual (mag)')
        ax6.set_title(f'Fourier Series Residuals (RMS={np.std(residual):.4f})')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "comparison_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存对比分析图: {output_path}")


def main():
    """主程序"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_dir, "TestData", "variable_Star", "delta_CEP_last12cycles_data.txt")
    output_dir = script_dir
    
    print(f"数据路径: {data_path}")
    print(f"输出路径: {output_dir}")
    
    processor = VariableStarSonificationV2(duration=30, samplerate=44100)
    time, mag = processor.load_data(data_path)
    
    # 方案1
    print("\n" + "=" * 60)
    print("方案1: 离散处理")
    print("=" * 60)
    
    processor.scheme1_discrete_time_domain(
        time, mag,
        os.path.join(output_dir, "scheme1_discrete_time.wav")
    )
    
    processor.scheme1_discrete_frequency_domain(
        time, mag,
        os.path.join(output_dir, "scheme1_discrete_freq.wav")
    )
    
    # 方案2
    print("\n" + "=" * 60)
    print("方案2: 连续处理（高精度）")
    print("=" * 60)
    
    processor.scheme2_continuous_time_domain(
        time, mag,
        os.path.join(output_dir, "scheme2_continuous_time.wav")
    )
    
    processor.scheme2_continuous_frequency_domain(
        time, mag,
        os.path.join(output_dir, "scheme2_continuous_freq.wav")
    )
    
    processor.scheme2_continuous_frequency_domain(
        time, mag,
        os.path.join(output_dir, "scheme2_continuous_freq.wav")
    )
    
    # 可视化
    processor.visualize_results(time, mag, output_dir)
    
    print("\n" + "=" * 60)
    print("所有处理完成!")
    print("=" * 60)
    print("\n输出文件:")
    print("  - scheme1_discrete_time.wav")
    print("  - scheme1_discrete_freq.wav")
    print("  - scheme2_continuous_time.wav")
    print("  - scheme2_continuous_freq.wav")
    print("  - comparison_analysis.png")


if __name__ == "__main__":
    main()
