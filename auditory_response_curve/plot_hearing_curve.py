"""
人耳频率响应曲线（等响曲线 / Equal Loudness Contour）
基于 ISO 226:2023 标准数据（使用 pydsm 库计算）
用于 "listen_to_the_universe" 天文可听化项目
"""

import numpy as np
import matplotlib.pyplot as plt
from pydsm import iso226

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# ISO 226:2023 等响曲线数据（来自 pydsm 库）
# =============================================================================

# 获取频率点
frequencies = iso226.tabled_f()

# 获取各phon值的等响曲线
phon_curves = {}
for phon in [20, 40, 60, 80, 100]:
    phon_curves[phon] = iso226.tabled_L_p(phon)

# 获取听力阈值
threshold_data = iso226.tabled_T_f()

print("数据加载完成！")
print(f"频率点数量: {len(frequencies)}")
print(f"频率范围: {frequencies[0]} Hz - {frequencies[-1]} Hz")

# =============================================================================
# 创建图形
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 9), dpi=150)

# 颜色映射（从深到浅）
colors = ['#1a5276', '#2874a6', '#3498db', '#5dade2', '#85c1e9']
phon_values = [20, 40, 60, 80, 100]

# 绘制等响曲线
for i, phon in enumerate(phon_values):
    spl_values = phon_curves[phon]
    ax.semilogx(frequencies, spl_values, color=colors[i], linewidth=2.5,
                label=f'{phon} phon 等响线', marker='o', markersize=4,
                markerfacecolor='white', markeredgecolor=colors[i], markeredgewidth=1.5)

# 绘制听力阈值曲线
ax.semilogx(frequencies, threshold_data, color='#e74c3c', linewidth=3,
            label='可听阈 (Hearing Threshold)', linestyle='--', marker='s',
            markersize=5, markerfacecolor='#e74c3c', markeredgecolor='white')

# 绘制痛觉阈值线
ax.semilogx(frequencies, np.ones_like(frequencies) * 120, color='#c0392b',
            linewidth=2, linestyle=':', label='痛觉阈值 (~120 dB)', alpha=0.7)

# 绘制安全聆听上限
ax.semilogx(frequencies, np.ones_like(frequencies) * 85, color='#f39c12',
            linewidth=2, linestyle='-.', label='安全上限 (85 dB)', alpha=0.8)

# 添加参考频率（1kHz）的标注
ax.axvline(x=1000, color='gray', linestyle=':', alpha=0.5)
ax.annotate('参考频率\n1 kHz', xy=(1000, -5), fontsize=9, ha='center', color='gray')

# 添加人耳最敏感区域标注（2-4kHz）- 曲线最低处
ax.axvspan(2000, 4000, alpha=0.15, color='green', label='最敏感区域 (2-4 kHz)')

# 标注关键频率点
key_freqs = [100, 1000, 4000, 8000]
for freq in key_freqs:
    ax.axvline(x=freq, color='gray', linestyle=':', alpha=0.3)

# 设置坐标轴
ax.set_xlim(20, 12500)
ax.set_ylim(-10, 140)
ax.set_xlabel('频率 (Hz)', fontsize=14, fontweight='bold')
ax.set_ylabel('声压级 (dB SPL)', fontsize=14, fontweight='bold')
ax.set_title('人耳频率响应曲线（等响曲线 / ISO 226:2023）\nHearing Frequency Response (Equal Loudness Contours)',
             fontsize=16, fontweight='bold', pad=15)

# 设置x轴刻度
ax.set_xticks([20, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 12500])
ax.set_xticklabels(['20', '31.5', '63', '125', '250', '500', '1k', '2k', '4k', '8k', '12.5k'])

# 设置y轴刻度
ax.set_yticks(range(0, 140, 10))

# 添加网格
ax.grid(True, which='major', linestyle='-', alpha=0.3)
ax.grid(True, which='minor', linestyle=':', alpha=0.15)

# 添加区域标注
ax.fill_between([20, 12500], 85, 130, alpha=0.08, color='red')
ax.text(25, 90, '危险区域 (>85dB)', fontsize=9, color='#c0392b', alpha=0.8)

# 添加图例
ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='gray')

# 添加注释框
textstr = '''关键发现：
• 1 kHz 是参考频率（phon值 = dB值）
• 2-4 kHz 人耳最敏感（曲线最低处）
• 低频(<100Hz)需要更高dB才能达到相同响度
• 高频(>8kHz)敏感度再次下降
• 视障青少年项目建议使用 500-4,000 Hz'''
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left', bbox=props)

plt.tight_layout()

# 保存图片
output_path = 'F:/My_github/listen_to_the_universe/auditory_response_curve/hearing_frequency_response.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'图片已保存至: {output_path}')

# 同时保存PDF矢量版本
pdf_path = 'F:/My_github/listen_to_the_universe/auditory_response_curve/hearing_frequency_response.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'PDF已保存至: {pdf_path}')

plt.show()
print('绘制完成！')
