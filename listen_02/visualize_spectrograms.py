"""
RGBL四通道音频可视化
生成频谱图和频率变化曲线
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal
import os

def plot_spectrogram(audio, samplerate, title, output_path, freq_range=None):
    """绘制频谱图"""
    plt.figure(figsize=(12, 6))
    
    # 计算频谱图
    f, t, Sxx = signal.spectrogram(audio, samplerate, nperseg=2048, noverlap=1024)
    
    # 绘制
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.colorbar(label='Intensity (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    
    if freq_range:
        plt.ylim(freq_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存频谱图: {output_path}")

def plot_frequency_trajectory(freq_history, duration, freq_range, title, output_path):
    """绘制频率随时间变化曲线"""
    plt.figure(figsize=(12, 4))
    
    time_axis = np.linspace(0, duration, len(freq_history))
    
    plt.plot(time_axis, freq_history, linewidth=1, alpha=0.8)
    plt.fill_between(time_axis, freq_history, alpha=0.3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if freq_range:
        plt.ylim(freq_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存频率轨迹: {output_path}")

def create_comparison_plot(output_dir, samplerate=44100, duration=20):
    """创建四个通道的对比图"""
    
    channels = ['L', 'R', 'G', 'B']
    colors = {'L': '#808080', 'R': '#e74c3c', 'G': '#27ae60', 'B': '#3498db'}
    freq_ranges = {
        'L': (200, 1200),
        'R': (1200, 2800),
        'G': (2800, 4800),
        'B': (4800, 8000)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('M38 RGBL Four-Channel Audio Spectrograms', fontsize=16, fontweight='bold')
    
    for idx, ch in enumerate(channels):
        ax = axes[idx // 2, idx % 2]
        
        wav_path = os.path.join(output_dir, f"M38_{ch}_channel.wav")
        if not os.path.exists(wav_path):
            continue
        
        sr, audio = wavfile.read(wav_path)
        
        # 计算频谱图
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=2048, noverlap=1024)
        
        # 绘制
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                          shading='gouraud', cmap='viridis')
        ax.set_ylim(freq_ranges[ch])
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{ch} Channel ({freq_ranges[ch][0]}-{freq_ranges[ch][1]} Hz)', 
                    color=colors[ch], fontweight='bold')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='dB')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "M38_RGBL_spectrograms_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存对比图: {output_path}")

def create_mixed_spectrogram(output_dir, samplerate=44100):
    """创建合成音频的频谱图"""
    
    wav_path = os.path.join(output_dir, "M38_RGBL_mixed.wav")
    if not os.path.exists(wav_path):
        return
    
    sr, audio = wavfile.read(wav_path)
    
    plt.figure(figsize=(14, 8))
    
    # 计算频谱图
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=4096, noverlap=2048)
    
    # 绘制
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                  shading='gouraud', cmap='plasma')
    plt.colorbar(label='Intensity (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('M38 RGBL Mixed Audio Spectrogram (Full Range)', fontsize=14, fontweight='bold')
    plt.ylim(0, 8000)
    
    # 标注四个频段
    plt.axhspan(200, 1200, alpha=0.1, color='gray', label='L (200-1200Hz)')
    plt.axhspan(1200, 2800, alpha=0.1, color='red', label='R (1200-2800Hz)')
    plt.axhspan(2800, 4800, alpha=0.1, color='green', label='G (2800-4800Hz)')
    plt.axhspan(4800, 8000, alpha=0.1, color='blue', label='B (4800-8000Hz)')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "M38_RGBL_mixed_spectrogram.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存合成频谱图: {output_path}")

def main():
    """主程序"""
    # 获取当前脚本所在目录（listen_02）
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("生成RGBL四通道音频可视化")
    print(f"工作目录: {output_dir}")
    print("=" * 60)
    
    # 创建对比图
    print("\n[生成四通道对比频谱图]")
    create_comparison_plot(output_dir)
    
    # 创建合成音频频谱图
    print("\n[生成合成音频频谱图]")
    create_mixed_spectrogram(output_dir)
    
    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
