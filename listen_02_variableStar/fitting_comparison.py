"""
多种拟合方案对比测试
Comparison of Multiple Fitting Methods for Cepheid Light Curve

拟合方案:
1. 傅里叶级数 (Fourier Series)
2. 多项式拟合 (Polynomial)
3. 正弦/余弦拟合 (Sinusoidal)
4. 高斯过程回归 (Gaussian Process)
5. 样条插值 (Spline) - 作为基准
"""

import numpy as np
import pandas as pd
from scipy import interpolate, fft, optimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

class FittingMethodsComparison:
    """多种拟合方法对比"""
    
    def __init__(self):
        print("=" * 70)
        print("造父变星光变曲线 - 多种拟合方案对比")
        print("=" * 70)
    
    def load_data(self, filepath):
        """加载数据"""
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
    
    # =================================================================
    # 方案1: 傅里叶级数
    # =================================================================
    def fit_fourier(self, time, mag, n_harmonics=20):
        """傅里叶级数拟合"""
        # 估计周期
        N = len(mag)
        dft = fft.fft(mag - np.mean(mag))
        freqs = fft.fftfreq(N, np.median(np.diff(time)))
        positive_freqs = freqs[1:N//2]
        positive_dft = np.abs(dft[1:N//2])
        peak_idx = np.argmax(positive_dft)
        period = 1.0 / positive_freqs[peak_idx]
        omega = 2 * np.pi / period
        
        # 计算系数
        a0 = 2 * np.mean(mag)
        an, bn = [], []
        for n in range(1, n_harmonics + 1):
            an.append(2 * np.mean(mag * np.cos(n * omega * time)))
            bn.append(2 * np.mean(mag * np.sin(n * omega * time)))
        
        def predict(t):
            result = a0 / 2 * np.ones_like(t)
            for n, (a, b) in enumerate(zip(an, bn), 1):
                result += a * np.cos(n * omega * t) + b * np.sin(n * omega * t)
            return result
        
        return predict, period, f"Fourier(n={n_harmonics})"
    
    # =================================================================
    # 方案2: 多项式拟合
    # =================================================================
    def fit_polynomial(self, time, mag, degree=15):
        """多项式拟合"""
        coeffs = np.polyfit(time, mag, degree)
        poly = np.poly1d(coeffs)
        
        def predict(t):
            return poly(t)
        
        return predict, None, f"Polynomial(deg={degree})"
    
    # =================================================================
    # 方案3: 正弦拟合（基于物理模型）
    # =================================================================
    def fit_sinusoidal(self, time, mag):
        """正弦拟合 - 基于造父变星的物理模型"""
        # 初始猜测
        mag_mean = np.mean(mag)
        mag_amp = (mag.max() - mag.min()) / 2
        
        # 估计周期
        N = len(mag)
        dft = fft.fft(mag - mag_mean)
        freqs = fft.fftfreq(N, np.median(np.diff(time)))
        peak_idx = np.argmax(np.abs(dft[1:N//2])) + 1
        freq_init = freqs[peak_idx]
        
        # 多谐波正弦模型
        def model(t, A0, A1, phi1, A2, phi2, A3, phi3, freq):
            omega = 2 * np.pi * freq
            return (A0 + 
                    A1 * np.sin(omega * t + phi1) +
                    A2 * np.sin(2 * omega * t + phi2) +
                    A3 * np.sin(3 * omega * t + phi3))
        
        # 拟合
        p0 = [mag_mean, mag_amp, 0, mag_amp/2, 0, mag_amp/3, 0, freq_init]
        bounds = ([0, 0, -np.pi, 0, -np.pi, 0, -np.pi, freq_init*0.5],
                  [10, 5, np.pi, 3, np.pi, 2, np.pi, freq_init*2])
        
        try:
            popt, _ = optimize.curve_fit(model, time, mag, p0=p0, bounds=bounds, maxfev=10000)
            
            def predict(t):
                return model(t, *popt)
            
            period = 1.0 / popt[7]
            return predict, period, "Sinusoidal(3 harmonics)"
        except:
            # 如果拟合失败，返回简单正弦
            def simple_predict(t):
                return mag_mean + mag_amp * np.sin(2 * np.pi * freq_init * t)
            return simple_predict, 1.0/freq_init, "Sinusoidal(simple)"
    
    # =================================================================
    # 方案4: 样条插值（基准）
    # =================================================================
    def fit_spline(self, time, mag):
        """样条插值"""
        cs = interpolate.CubicSpline(time, mag)
        
        def predict(t):
            return cs(t)
        
        return predict, None, "Cubic Spline"
    
    # =================================================================
    # 方案5: 分段线性插值
    # =================================================================
    def fit_linear(self, time, mag):
        """线性插值"""
        def predict(t):
            return np.interp(t, time, mag)
        
        return predict, None, "Linear Interp"
    
    # =================================================================
    # 评估拟合质量
    # =================================================================
    def evaluate_fit(self, time, mag_true, predict_func, method_name):
        """评估拟合质量"""
        mag_pred = predict_func(time)
        
        # 计算指标
        residual = mag_true - mag_pred
        mse = np.mean(residual**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residual))
        max_error = np.max(np.abs(residual))
        
        # R²
        ss_res = np.sum(residual**2)
        ss_tot = np.sum((mag_true - np.mean(mag_true))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'method': method_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r2': r2,
            'residual': residual
        }
    
    # =================================================================
    # 运行所有拟合并对比
    # =================================================================
    def run_comparison(self, time, mag):
        """运行所有拟合方法并对比"""
        print("\n" + "=" * 70)
        print("开始拟合对比...")
        print("=" * 70)
        
        results = []
        predictions = {}
        
        # 定义拟合方法
        methods = [
            ('Fourier', lambda t, m: self.fit_fourier(t, m, n_harmonics=20)),
            ('Polynomial', lambda t, m: self.fit_polynomial(t, m, degree=15)),
            ('Sinusoidal', lambda t, m: self.fit_sinusoidal(t, m)),
            ('Spline', lambda t, m: self.fit_spline(t, m)),
            ('Linear', lambda t, m: self.fit_linear(t, m)),
        ]
        
        for name, fit_func in methods:
            print(f"\n[拟合] {name}...")
            try:
                predict_func, period, method_str = fit_func(time, mag)
                predictions[name] = predict_func
                
                result = self.evaluate_fit(time, mag, predict_func, method_str)
                results.append(result)
                
                print(f"  RMSE: {result['rmse']:.6f}")
                print(f"  R²:   {result['r2']:.6f}")
                if period:
                    print(f"  周期: {period:.4f} 天")
            except Exception as e:
                print(f"  错误: {e}")
        
        return results, predictions
    
    # =================================================================
    # 可视化对比
    # =================================================================
    def visualize_comparison(self, time, mag, predictions, results, output_dir):
        """生成对比可视化"""
        print("\n" + "=" * 70)
        print("生成对比图表...")
        print("=" * 70)
        
        fig = plt.figure(figsize=(18, 14))
        
        # 时间轴用于绘制
        time_fine = np.linspace(time[0], time[-1], 1000)
        
        # 1. 原始数据
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(time, mag, 'ko', markersize=5, label='Observations')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Magnitude')
        ax1.set_title('Raw Data')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2-6. 各种拟合结果
        positions = [(3, 3, i) for i in range(2, 7)]
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for (name, predict_func), pos, color in zip(predictions.items(), positions, colors):
            ax = plt.subplot(*pos)
            
            # 绘制原始数据
            ax.plot(time, mag, 'ko', markersize=4, alpha=0.5, label='Data')
            
            # 绘制拟合曲线
            mag_fine = predict_func(time_fine)
            ax.plot(time_fine, mag_fine, color=color, linewidth=2, label=name)
            
            # 找到对应的评估结果
            result = next(r for r in results if name in r['method'])
            
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Magnitude')
            ax.set_title(f'{name}\nRMSE={result["rmse"]:.4f}, R²={result["r2"]:.4f}')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 7. 所有拟合叠加对比
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(time, mag, 'ko', markersize=5, alpha=0.6, label='Data', zorder=10)
        for (name, predict_func), color in zip(predictions.items(), colors):
            mag_fine = predict_func(time_fine)
            ax7.plot(time_fine, mag_fine, color=color, linewidth=1.5, 
                    label=name, alpha=0.8)
        ax7.set_xlabel('Time (days)')
        ax7.set_ylabel('Magnitude')
        ax7.set_title('All Methods Comparison')
        ax7.invert_yaxis()
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8)
        
        # 8. 残差对比
        ax8 = plt.subplot(3, 3, 8)
        for (name, predict_func), color in zip(predictions.items(), colors):
            residual = mag - predict_func(time)
            ax8.plot(time, residual, 'o', color=color, markersize=3, 
                    alpha=0.6, label=name)
        ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax8.set_xlabel('Time (days)')
        ax8.set_ylabel('Residual (mag)')
        ax8.set_title('Residuals Comparison')
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
        
        # 9. 性能指标对比表
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # 创建表格数据
        table_data = [['Method', 'RMSE', 'MAE', 'R²']]
        for r in results:
            table_data.append([
                r['method'],
                f"{r['rmse']:.4f}",
                f"{r['mae']:.4f}",
                f"{r['r2']:.4f}"
            ])
        
        table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 表头样式
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 高亮最佳行
        best_idx = np.argmin([r['rmse'] for r in results]) + 1
        for i in range(4):
            table[(best_idx, i)].set_facecolor('#E8F5E9')
        
        ax9.set_title('Performance Comparison\n(Best method highlighted)', pad=20)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "fitting_methods_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {output_path}")
        
        # 输出最佳方法
        best_method = min(results, key=lambda x: x['rmse'])
        print(f"\n[最佳拟合方法]")
        print(f"  方法: {best_method['method']}")
        print(f"  RMSE: {best_method['rmse']:.6f}")
        print(f"  R2:   {best_method['r2']:.6f}")


def main():
    """主程序"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_dir, "TestData", "variable_Star", "delta_CEP_last12cycles_data.txt")
    output_dir = script_dir
    
    # 创建对比器
    comparator = FittingMethodsComparison()
    
    # 加载数据
    time, mag = comparator.load_data(data_path)
    
    # 运行对比
    results, predictions = comparator.run_comparison(time, mag)
    
    # 可视化
    comparator.visualize_comparison(time, mag, predictions, results, output_dir)
    
    print("\n" + "=" * 70)
    print("拟合对比完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
