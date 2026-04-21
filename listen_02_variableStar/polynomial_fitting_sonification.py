"""
多项式拟合光变曲线 + 音频化
Polynomial Fitting for Cepheid Light Curve Sonification

核心思路：
1. 使用多项式拟合光变曲线（无需周期性假设）
2. 对比不同阶数的多项式拟合效果
3. 选择最佳拟合生成连续音高音频
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fft
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PolynomialFittingSonification:
    """多项式拟合 + 音频化"""
    
    # 音频参数
    SAMPLE_RATE = 44100
    DURATION = 8.0
    FREQ_MIN = 200
    FREQ_MAX = 800
    
    def __init__(self):
        print("=" * 70)
        print("多项式拟合光变曲线音频化")
        print("Polynomial Fitting Sonification")
        print("=" * 70)
    
    def load_data(self, filepath):
        """加载变星光变数据"""
        df = pd.read_csv(filepath, sep='\t', skipinitialspace=True)
        jd = df['JD'].values
        mag = df['Magnitude'].values
        
        # 排序并去重
        sort_idx = np.argsort(jd)
        jd, mag = jd[sort_idx], mag[sort_idx]
        
        unique_jd, unique_mag = [], []
        for i in range(len(jd)):
            if i == 0 or not np.isclose(jd[i], jd[i-1], rtol=1e-10):
                unique_jd.append(jd[i])
                unique_mag.append(mag[i])
            else:
                unique_mag[-1] = (unique_mag[-1] + mag[i]) / 2
        
        time = np.array(unique_jd) - unique_jd[0]
        mag = np.array(unique_mag)
        
        print(f"\n[数据信息]")
        print(f"  观测点数: {len(time)}")
        print(f"  时间跨度: {time[-1]:.2f} 天")
        print(f"  星等范围: [{mag.min():.3f}, {mag.max():.3f}]")
        
        return time, mag
    
    def polynomial_fit(self, time, mag, degree):
        """
        多项式拟合
        
        Parameters:
            time: 时间数组
            mag: 星等数组
            degree: 多项式阶数
        
        Returns:
            coeffs: 多项式系数
            poly_func: 拟合函数
            rmse: 均方根误差
        """
        # 归一化时间到[0, 1]区间，提高数值稳定性
        t_normalized = (time - time[0]) / (time[-1] - time[0])
        
        # 多项式拟合
        coeffs = np.polyfit(t_normalized, mag, degree)
        poly = np.poly1d(coeffs)
        
        # 预测值
        mag_pred = poly(t_normalized)
        
        # 计算RMSE
        rmse = np.sqrt(np.mean((mag - mag_pred) ** 2))
        
        # 创建预测函数（接受原始时间）
        def poly_func(t):
            t_norm = (t - time[0]) / (time[-1] - time[0])
            return poly(t_norm)
        
        return coeffs, poly_func, rmse
    
    def find_optimal_degree(self, time, mag, max_degree=25):
        """
        寻找最优多项式阶数
        
        策略：
        1. 尝试不同阶数
        2. 使用肘部法则找到RMSE下降趋于平缓的点
        3. 避免过拟合（高阶多项式振荡）
        4. 限制在合理范围内（建议8-15阶）
        """
        print("\n" + "=" * 70)
        print("寻找最优多项式阶数...")
        print("=" * 70)
        
        degrees = list(range(3, max_degree + 1, 2))
        results = []
        
        for deg in degrees:
            _, _, rmse = self.polynomial_fit(time, mag, deg)
            results.append({'degree': deg, 'rmse': rmse})
            print(f"  阶数 {deg:2d}: RMSE = {rmse:.6f}")
        
        # 肘部法则：找到RMSE下降趋于平缓的点
        rmses = [r['rmse'] for r in results]
        improvements = np.diff(rmses)
        
        elbow_deg = degrees[-1]
        if len(improvements) > 1:
            second_deriv = np.diff(improvements)
            elbow_idx = np.argmax(second_deriv) + 1
            elbow_deg = degrees[elbow_idx]
        
        print(f"\n  肘部法则推荐阶数: {elbow_deg}")
        
        # 最优阶数：肘部点与上限的较小值
        # 限制在15阶以内，避免过拟合振荡
        MAX_RECOMMENDED_DEG = 15
        optimal_deg = min(elbow_deg, MAX_RECOMMENDED_DEG)
        
        # 如果肘部点太低（<8），选择8
        MIN_RECOMMENDED_DEG = 8
        optimal_deg = max(optimal_deg, MIN_RECOMMENDED_DEG)
        
        # 找到最接近optimal_deg的奇数/偶数
        if optimal_deg % 2 == 0:
            optimal_deg += 1
        
        print(f"  最终推荐阶数: {optimal_deg} (范围: {MIN_RECOMMENDED_DEG}-{MAX_RECOMMENDED_DEG})")
        print(f"  说明: 阶数过低拟合不足，过高会导致边界振荡")
        
        return optimal_deg, results
    
    def compare_degrees(self, time, mag, degrees=[5, 10, 15, 20, 25], output_dir=None):
        """可视化对比不同阶数的拟合效果"""
        print("\n" + "=" * 70)
        print("对比不同阶数的拟合效果...")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        t_normalized = np.linspace(0, 1, 1000)
        t_fine = time[0] + t_normalized * (time[-1] - time[0])
        
        results = []
        
        for i, deg in enumerate(degrees):
            ax = axes[i]
            
            # 拟合
            coeffs, poly_func, rmse = self.polynomial_fit(time, mag, deg)
            results.append({'degree': deg, 'rmse': rmse, 'poly_func': poly_func})
            
            # 预测
            mag_fine = poly_func(t_fine)
            mag_pred = poly_func(time)
            
            # 绘制
            ax.plot(time, mag, 'ko', markersize=5, alpha=0.6, label='观测数据')
            ax.plot(t_fine, mag_fine, 'r-', linewidth=2, label=f'多项式拟合 (deg={deg})')
            ax.plot(time, mag_pred, 'bx', markersize=4, alpha=0.5, label='拟合点')
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'Polynomial Degree {deg}\nRMSE = {rmse:.6f}')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # 第6个子图：RMSE随阶数变化
        ax6 = axes[5]
        degs = [r['degree'] for r in results]
        rmses = [r['rmse'] for r in results]
        ax6.plot(degs, rmses, 'bo-', linewidth=2, markersize=8)
        ax6.set_xlabel('Polynomial Degree')
        ax6.set_ylabel('RMSE')
        ax6.set_title('RMSE vs Polynomial Degree')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = os.path.join(output_dir, "polynomial_degree_comparison.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  已保存: {output_path}")
        
        plt.close()
        
        return results
    
    def generate_audio(self, time, mag, poly_func, output_path):
        """
        基于多项式拟合生成音频（连续频率变化）
        
        星等越大(越暗) → 频率越低
        星等越小(越亮) → 频率越高
        """
        print(f"\n[生成音频] {output_path}")
        
        # 生成时间轴
        t_audio = np.linspace(0, self.DURATION, int(self.SAMPLE_RATE * self.DURATION))
        
        # 将音频时间映射到光变时间
        t_normalized = t_audio / self.DURATION
        t_curve = time[0] + t_normalized * (time[-1] - time[0])
        
        # 使用多项式拟合计算星等
        mag_curve = poly_func(t_curve)
        
        # 限制星等范围（防止过拟合导致的极值）
        mag_min_orig, mag_max_orig = mag.min(), mag.max()
        mag_safe_min = mag_min_orig - 0.5  # 允许稍微超出观测范围
        mag_safe_max = mag_max_orig + 0.5
        mag_curve_clipped = np.clip(mag_curve, mag_safe_min, mag_safe_max)
        
        # 星等 → 频率映射
        freq_curve = self.FREQ_MAX - (mag_curve_clipped - mag_min_orig) / (mag_max_orig - mag_min_orig) * (self.FREQ_MAX - self.FREQ_MIN)
        
        # 确保频率在有效范围内
        freq_curve = np.clip(freq_curve, self.FREQ_MIN, self.FREQ_MAX)
        
        # 使用Chirp信号：频率随时间连续变化
        # 计算相位积分
        phase = 2 * np.pi * np.cumsum(freq_curve) / self.SAMPLE_RATE
        audio = np.sin(phase)
        
        # 添加幅度包络（可选）
        # 亮星（低星等）声音更响亮
        amplitude = 1.0 - 0.3 * (mag_curve_clipped - mag_min_orig) / (mag_max_orig - mag_min_orig)
        audio = audio * amplitude
        
        # 归一化
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # 转换为16位整数
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 保存
        wavfile.write(output_path, self.SAMPLE_RATE, audio_int16)
        print(f"  音频时长: {self.DURATION}s")
        print(f"  频率范围: {freq_curve.min():.1f} - {freq_curve.max():.1f} Hz")
        print(f"  星等范围(裁剪后): {mag_curve_clipped.min():.3f} - {mag_curve_clipped.max():.3f}")
        
        return output_path
    
    def generate_freq_domain_audio(self, time, mag, poly_func, output_path):
        """
        生成频域表示音频（频谱扫描听化）
        
        核心思路：
        - 对拟合曲线做FFT得到频谱
        - 从低频到高频依次"播放"每个频率分量
        - 音频时间轴 = 频谱频率轴（从低频到高频扫描）
        - 音频响度/音高 = 频谱幅度大小
        """
        print(f"\n[生成频域音频] {output_path}")
        
        # 使用拟合后的高密度数据计算频谱
        N_fft = 8192
        t_fine = np.linspace(time[0], time[-1], N_fft)
        mag_fitted = poly_func(t_fine)
        
        # 去均值并加窗
        mag_detrended = mag_fitted - np.mean(mag_fitted)
        window = np.hanning(N_fft)
        mag_windowed = mag_detrended * window
        
        # FFT
        dft = fft.fft(mag_windowed)
        freqs = fft.fftfreq(N_fft, np.median(np.diff(t_fine)))
        
        # 只取正频率部分
        positive_mask = freqs > 0
        positive_freqs = freqs[positive_mask]
        positive_dft = np.abs(dft[positive_mask])
        
        # 只取有意义的低频部分（前500个频点，排除极高频噪声）
        n_freq_points = min(500, len(positive_freqs) // 4)
        spectrum_freqs = positive_freqs[:n_freq_points]
        spectrum_mags = positive_dft[:n_freq_points]
        
        print(f"  FFT点数: {N_fft}")
        print(f"  频谱范围: {spectrum_freqs[0]:.6f} - {spectrum_freqs[-1]:.6f} cycles/day")
        print(f"  频谱分量数: {n_freq_points}")
        
        # 生成音频：从低频到高频扫描播放频谱
        # 每个频点分配相同的播放时间
        segment_duration = self.DURATION / n_freq_points
        samples_per_segment = int(segment_duration * self.SAMPLE_RATE)
        
        audio_segments = []
        
        for i in range(n_freq_points):
            # 频谱幅度归一化（用于控制音量）
            magnitude_norm = spectrum_mags[i] / (np.max(spectrum_mags) + 1e-10)
            
            # 方案：响度表征幅度大小
            # 使用固定音高（中频），音量随频谱幅度变化
            # 或者：音高从低到高扫描，同时音量表征幅度
            
            # 这里采用：音高从低到高扫描 + 音量表征幅度
            progress = i / n_freq_points  # 0 -> 1
            audio_freq = self.FREQ_MIN + progress * (self.FREQ_MAX - self.FREQ_MIN)
            
            # 生成该频点的声音
            t_segment = np.linspace(0, segment_duration, samples_per_segment)
            
            # 音量与频谱幅度成正比（最小0.1保证能听到）
            amplitude = 0.1 + 0.9 * magnitude_norm
            
            # 使用纯音，频率从低到高
            segment = amplitude * np.sin(2 * np.pi * audio_freq * t_segment)
            
            # 添加淡入淡出，避免咔嗒声
            fade_samples = min(50, samples_per_segment // 5)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                segment[:fade_samples] *= fade_in
                segment[-fade_samples:] *= fade_out
            
            audio_segments.append(segment)
        
        # 合并所有段
        audio = np.concatenate(audio_segments)
        
        # 确保长度正确
        target_samples = int(self.SAMPLE_RATE * self.DURATION)
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        
        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.9
        
        # 转换为16位整数
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_path, self.SAMPLE_RATE, audio_int16)
        
        print(f"  音频时长: {self.DURATION}s")
        print(f"  扫描范围: {self.FREQ_MIN} - {self.FREQ_MAX} Hz")
        print(f"  每段时长: {segment_duration*1000:.1f} ms")
        
        return output_path
    
    def visualize_fitting(self, time, mag, poly_func, degree, rmse, output_dir):
        """可视化拟合结果和音频映射"""
        print("\n" + "=" * 70)
        print("生成可视化图表...")
        print("=" * 70)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 时间轴
        t_normalized = np.linspace(0, 1, 1000)
        t_fine = time[0] + t_normalized * (time[-1] - time[0])
        mag_fine = poly_func(t_fine)
        
        # 1. 原始数据与拟合曲线
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(time, mag, 'ko', markersize=6, alpha=0.6, label='观测数据')
        ax1.plot(t_fine, mag_fine, 'r-', linewidth=2, label=f'多项式拟合 (deg={degree})')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'Polynomial Fitting\nRMSE = {rmse:.6f}')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 残差分析
        mag_pred = poly_func(time)
        residual = mag - mag_pred
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(time, residual, 'bo', markersize=4, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Residual (mag)')
        ax2.set_title('Fitting Residuals')
        ax2.grid(True, alpha=0.3)
        
        # 3. 残差直方图
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(residual, bins=20, color='blue', alpha=0.6, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Residual (mag)')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Residual Distribution\nStd = {np.std(residual):.4f}')
        ax3.grid(True, alpha=0.3)
        
        # 4. 星等-频率映射
        mag_min, mag_max = mag.min(), mag.max()
        freq_fine = self.FREQ_MAX - (mag_fine - mag_min) / (mag_max - mag_min) * (self.FREQ_MAX - self.FREQ_MIN)
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(t_fine, freq_fine, 'g-', linewidth=2)
        ax4.set_xlabel('Time (days)')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_title(f'Magnitude → Frequency Mapping\n[{self.FREQ_MIN}-{self.FREQ_MAX} Hz]')
        ax4.grid(True, alpha=0.3)
        
        # 5. 频率随时间变化
        ax5 = plt.subplot(3, 3, 5)
        t_audio = np.linspace(0, self.DURATION, 1000)
        ax5.plot(t_audio, freq_fine, 'm-', linewidth=2)
        ax5.set_xlabel('Audio Time (s)')
        ax5.set_ylabel('Frequency (Hz)')
        ax5.set_title('Audio Frequency Envelope')
        ax5.grid(True, alpha=0.3)
        
        # 6. 音频波形预览
        ax6 = plt.subplot(3, 3, 6)
        phase = 2 * np.pi * np.cumsum(freq_fine) / self.SAMPLE_RATE
        waveform = np.sin(phase)
        amplitude = 1.0 - 0.3 * (mag_fine - mag_min) / (mag_max - mag_min)
        waveform = waveform * amplitude
        ax6.plot(t_audio[:500], waveform[:500], 'b-', linewidth=0.5)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Amplitude')
        ax6.set_title('Audio Waveform (Preview)')
        ax6.grid(True, alpha=0.3)
        
        # 7. 拟合数据频谱
        ax7 = plt.subplot(3, 3, 7)
        mag_detrended = mag_fine - np.mean(mag_fine)
        dft = fft.fft(mag_detrended)
        freqs = fft.fftfreq(len(mag_fine), np.median(np.diff(t_fine)))
        positive_mask = freqs > 0
        ax7.semilogy(freqs[positive_mask], np.abs(dft[positive_mask]), 'b-', linewidth=1)
        ax7.set_xlabel('Frequency (cycles/day)')
        ax7.set_ylabel('Magnitude')
        ax7.set_title('Spectrum of Fitted Curve')
        ax7.grid(True, alpha=0.3)
        
        # 8. 观测数据与拟合数据对比
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(time, mag, 'ko', markersize=6, alpha=0.6, label='观测数据')
        ax8.plot(time, mag_pred, 'rx', markersize=6, alpha=0.8, label='拟合值')
        ax8.set_xlabel('Time (days)')
        ax8.set_ylabel('Magnitude')
        ax8.set_title('Data vs Fitted Values')
        ax8.invert_yaxis()
        ax8.grid(True, alpha=0.3)
        ax8.legend()
        
        # 9. 多项式系数显示
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        coeffs, _, _ = self.polynomial_fit(time, mag, degree)
        coeff_text = f"Polynomial Coefficients (degree {degree})\n"
        coeff_text += "=" * 40 + "\n\n"
        coeff_text += "Highest degree first (numpy.polyfit format)\n\n"
        
        for i, c in enumerate(coeffs[:5]):
            coeff_text += f"  x^{degree-i}: {c:.6e}\n"
        if len(coeffs) > 5:
            coeff_text += f"  ... ({len(coeffs)-5} more coefficients)\n"
        
        coeff_text += f"\nFitting Quality:\n"
        coeff_text += f"  RMSE: {rmse:.6f}\n"
        coeff_text += f"  Max Error: {np.max(np.abs(residual)):.6f}\n"
        
        ax9.text(0.1, 0.9, coeff_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax9.set_title('Polynomial Coefficients')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"polynomial_fitting_deg{degree}_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {output_path}")
        
        return output_path
    
    def run(self, data_path, output_dir):
        """运行完整流程"""
        # 加载数据
        time, mag = self.load_data(data_path)
        
        # 寻找最优阶数
        optimal_deg, degree_results = self.find_optimal_degree(time, mag, max_degree=25)
        
        # 对比不同阶数
        degree_comparison = self.compare_degrees(time, mag, degrees=[5, 10, 15, 20, 25], output_dir=output_dir)
        
        # 使用最优阶数进行拟合
        print("\n" + "=" * 70)
        print(f"使用最优阶数 {optimal_deg} 进行最终拟合...")
        print("=" * 70)
        
        coeffs, poly_func, rmse = self.polynomial_fit(time, mag, optimal_deg)
        print(f"  多项式阶数: {optimal_deg}")
        print(f"  拟合RMSE: {rmse:.6f}")
        
        # 生成可视化
        self.visualize_fitting(time, mag, poly_func, optimal_deg, rmse, output_dir)
        
        # 生成音频 - 时域连续音高
        audio_path_time = os.path.join(output_dir, f"polynomial_time_deg{optimal_deg}.wav")
        self.generate_audio(time, mag, poly_func, audio_path_time)
        
        # 生成音频 - 频域表示
        audio_path_freq = os.path.join(output_dir, f"polynomial_freq_deg{optimal_deg}.wav")
        self.generate_freq_domain_audio(time, mag, poly_func, audio_path_freq)
        
        # 额外：生成几个不同阶数的音频供对比
        print("\n" + "=" * 70)
        print("生成不同阶数的音频文件供对比...")
        print("=" * 70)
        
        for deg in [10, 15, 20]:
            if deg != optimal_deg:
                _, pf, _ = self.polynomial_fit(time, mag, deg)
                self.generate_audio(time, mag, pf, os.path.join(output_dir, f"polynomial_time_deg{deg}.wav"))
        
        print("\n" + "=" * 70)
        print("多项式拟合音频化完成!")
        print("=" * 70)
        print(f"\n输出文件:")
        print(f"  - 阶数对比图: polynomial_degree_comparison.png")
        print(f"  - 详细分析图: polynomial_fitting_deg{optimal_deg}_analysis.png")
        print(f"  - 时域音频(deg{optimal_deg}): polynomial_time_deg{optimal_deg}.wav")
        print(f"  - 频域音频(deg{optimal_deg}): polynomial_freq_deg{optimal_deg}.wav")
        print(f"  - 其他阶数时域音频: polynomial_time_deg*.wav")


def main():
    """主程序"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_dir, "TestData", "variable_Star", "delta_CEP_last12cycles_data.txt")
    output_dir = script_dir
    
    # 创建处理器
    processor = PolynomialFittingSonification()
    processor.run(data_path, output_dir)


if __name__ == "__main__":
    main()
