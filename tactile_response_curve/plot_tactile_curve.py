"""
人体表面温度触觉响应曲线
Human Tactile Temperature Response Curve
用于 "listen_to_the_universe" 天文可触化项目
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 温度敏感度数据（基于 Stevens & Choo, 1994; Jones & Berris, 2002 等研究）
# =============================================================================

# 温度范围 (°C)
temperatures = np.linspace(5, 50, 100)

# 冷感受器活跃度（Krause小体，15-35°C 最活跃）
def cold_receptor_response(T):
    """冷感受器响应：15-35°C 最敏感，向两端递减"""
    # 使用高斯函数模拟，在 25°C 达到峰值
    sigma = 10
    mu = 25
    return 100 * np.exp(-((T - mu) ** 2) / (2 * sigma ** 2))

# 热感受器活跃度（Ruffini终球，30-45°C 最活跃）
def warm_receptor_response(T):
    """热感受器响应：30-45°C 最敏感，向两端递减"""
    sigma = 7.5
    mu = 37
    return 100 * np.exp(-((T - mu) ** 2) / (2 * sigma ** 2))

# 温度辨别阈值（最小可感知温度变化）
def discrimination_threshold(T):
    """温度辨别阈值：静态接触时约 0.5-1°C，快速扫描时 2-3°C"""
    if T < 20:
        return 2.5 - 0.05 * T  # 低温区阈值较高
    elif T < 30:
        return 1.0 + 0.05 * (30 - T)  # 舒适区阈值最低
    elif T < 38:
        return 0.8 + 0.1 * (T - 30)  # 略升温区
    else:
        return 1.5 + 0.15 * (T - 38)  # 高温区阈值上升

# 舒适度评分（0-1）
def comfort_score(T):
    """主观舒适度评分：28-34°C 最舒适"""
    if 28 <= T <= 34:
        return 1.0
    elif 25 <= T < 28 or 34 < T <= 36:
        return 0.7
    elif 20 <= T < 25 or 36 < T <= 38:
        return 0.3
    elif 15 <= T < 20 or 38 < T <= 40:
        return 0.1
    else:
        return 0.0

# =============================================================================
# 创建图形
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
fig.suptitle('人体表面温度触觉响应曲线\nHuman Tactile Temperature Response', 
             fontsize=16, fontweight='bold', y=0.98)

# -----------------------------------------------------------------------------
# 子图1：感受器响应曲线
# -----------------------------------------------------------------------------
ax1 = axes[0, 0]

cold_resp = cold_receptor_response(temperatures)
warm_resp = warm_receptor_response(temperatures)

ax1.plot(temperatures, cold_resp, 'b-', linewidth=2.5, label='冷感受器 (Krause小体)')
ax1.plot(temperatures, warm_resp, 'r-', linewidth=2.5, label='热感受器 (Ruffini终球)')

# 标注峰值
ax1.axvline(x=25, color='blue', linestyle=':', alpha=0.5)
ax1.axvline(x=37, color='red', linestyle=':', alpha=0.5)
ax1.annotate('冷感受器峰值\n~25°C', xy=(25, 100), xytext=(10, 85),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'), color='blue')
ax1.annotate('热感受器峰值\n~37°C', xy=(37, 100), xytext=(42, 85),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='red'), color='red')

ax1.set_xlim(5, 50)
ax1.set_ylim(0, 110)
ax1.set_xlabel('温度 (°C)', fontsize=12)
ax1.set_ylabel('感受器活跃度 (%)', fontsize=12)
ax1.set_title('温度感受器响应曲线', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.fill_betweenx([0, 110], 15, 40, alpha=0.1, color='green')

# -----------------------------------------------------------------------------
# 子图2：温度辨别阈值
# -----------------------------------------------------------------------------
ax2 = axes[0, 1]

thresholds = [discrimination_threshold(T) for T in temperatures]

ax2.plot(temperatures, thresholds, 'g-', linewidth=2.5)
ax2.fill_between(temperatures, thresholds, alpha=0.3, color='green')

# 标注舒适区
ax2.axvspan(28, 34, alpha=0.2, color='green', label='舒适区')
ax2.axvline(x=28, color='green', linestyle='--', alpha=0.7)
ax2.axvline(x=34, color='green', linestyle='--', alpha=0.7)

# 添加阈值参考线
ax2.axhline(y=1.0, color='orange', linestyle=':', label='日常感知阈值 (~1°C)')
ax2.axhline(y=0.5, color='red', linestyle=':', label='训练感知阈值 (~0.5°C)')

ax2.set_xlim(5, 50)
ax2.set_ylim(0, 5)
ax2.set_xlabel('温度 (°C)', fontsize=12)
ax2.set_ylabel('最小可辨别温差 (°C)', fontsize=12)
ax2.set_title('温度辨别阈值曲线', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# 子图3：舒适度评分
# -----------------------------------------------------------------------------
ax3 = axes[1, 0]

comforts = [comfort_score(T) for T in temperatures]

# 使用颜色映射：蓝色(冷) -> 绿色(舒适) -> 红色(热)
colors = []
for T in temperatures:
    if T < 15 or T > 45:
        colors.append('#8B0000')  # 深红（危险）
    elif T < 20 or T > 40:
        colors.append('#FF6347')  # 番茄红（警告）
    elif T < 28 or T > 36:
        colors.append('#FFD700')  # 金色（一般）
    else:
        colors.append('#32CD32')  # 绿色（舒适）
    if 25 <= T <= 38:
        colors[-1] = '#32CD32'  # 绿色（舒适）

ax3.bar(temperatures, comforts, width=0.5, color=colors, edgecolor='none', alpha=0.8)

# 标注区域
ax3.axhspan(0.7, 1.0, xmin=0.44, xmax=0.68, alpha=0.15, color='green')

ax3.set_xlim(5, 50)
ax3.set_ylim(0, 1.2)
ax3.set_xlabel('温度 (°C)', fontsize=12)
ax3.set_ylabel('舒适度评分', fontsize=12)
ax3.set_title('主观舒适度分布', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#32CD32', label='舒适 (28-34°C)'),
    Patch(facecolor='#FFD700', label='一般 (20-28°C, 34-38°C)'),
    Patch(facecolor='#FF6347', label='警告 (<20°C, >38°C)'),
    Patch(facecolor='#8B0000', label='危险 (<10°C, >45°C)'),
]
ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)

# -----------------------------------------------------------------------------
# 子图4：安全操作区间（综合视图）
# -----------------------------------------------------------------------------
ax4 = axes[1, 1]

# 创建背景区域
ax4.axvspan(10, 45, alpha=0.05, color='gray', label='理论可感知区间')
ax4.axvspan(15, 40, alpha=0.1, color='blue', label='安全接触区间')
ax4.axvspan(25, 38, alpha=0.2, color='green', label='推荐操作区间')

# 安全/危险边界
ax4.axvline(x=10, color='red', linewidth=2, linestyle='--', label='冻伤风险 (<10°C)')
ax4.axvline(x=45, color='darkred', linewidth=2, linestyle='--', label='烫伤风险 (>45°C)')
ax4.axvline(x=40, color='orange', linewidth=1.5, linestyle=':', label='高温警告 (>40°C)')

# 绘制感受器响应（双Y轴）
ax4_twin = ax4.twinx()
ax4_twin.plot(temperatures, cold_resp, 'b-', linewidth=2, alpha=0.7, label='冷感受器')
ax4_twin.plot(temperatures, warm_resp, 'r-', linewidth=2, alpha=0.7, label='热感受器')
ax4_twin.set_ylabel('感受器活跃度 (%)', fontsize=11, color='gray')
ax4_twin.tick_params(axis='y', labelcolor='gray')
ax4_twin.set_ylim(0, 110)

# 天文应用标注
ax4.annotate('← 冷星\n(红巨星)', xy=(25, 80), fontsize=9, ha='center', color='#1a5276')
ax4.annotate('热星 →\n(蓝巨星)', xy=(38, 80), fontsize=9, ha='center', color='#c0392b')

ax4.set_xlim(5, 50)
ax4.set_ylim(0, 100)
ax4.set_xlabel('温度 (°C)', fontsize=12)
ax4.set_ylabel('安全等级', fontsize=12)
ax4.set_title('安全操作区间与天文温度映射', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
output_path = 'F:/My_github/listen_to_the_universe/tactile_response_curve/tactile_temperature_response.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'图片已保存至: {output_path}')

# 同时保存PDF矢量版本
pdf_path = 'F:/My_github/listen_to_the_universe/tactile_response_curve/tactile_temperature_response.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'PDF已保存至: {pdf_path}')

plt.show()
print('绘制完成！')

# =============================================================================
# 单独绘制关键参数汇总图
# =============================================================================
fig2, ax = plt.subplots(figsize=(10, 4), dpi=150)
ax.axis('off')

# 表格数据
table_data = [
    ['参数类型', '数值', '说明'],
    ['安全接触范围', '25°C - 38°C', '对人体皮肤无伤害'],
    ['推荐操作范围', '28°C - 34°C', '最佳舒适区，阈值最低'],
    ['低温警告边界', '< 15°C', '开始有明显冷感'],
    ['高温警告边界', '> 40°C', '需严格控制时长'],
    ['绝对禁止', '< 10°C 或 > 45°C', '冻伤/烫伤风险'],
    ['静态辨别阈值', '0.5 - 1.0°C', '训练可达到 0.5°C'],
    ['快速扫描阈值', '2.0 - 3.0°C', '日常感知水平'],
    ['变化速率', '≤ 1°C/秒', '避免热冲击'],
    ['极限温度持续', '≤ 3 秒', '高温状态最长接触时间'],
]

# 创建表格
table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 cellLoc='center', loc='center',
                 colWidths=[0.3, 0.25, 0.45])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# 设置表头样式
for i in range(3):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

# 设置交替行颜色
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        else:
            table[(i, j)].set_facecolor('#ffffff')

ax.set_title('触觉温度设计关键参数汇总\nKey Parameters for Tactile Temperature Design',
            fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

# 保存参数汇总图
summary_path = 'F:/My_github/listen_to_the_universe/tactile_response_curve/tactile_parameters_summary.png'
plt.savefig(summary_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'参数汇总图已保存至: {summary_path}')

summary_pdf = 'F:/My_github/listen_to_the_universe/tactile_response_curve/tactile_parameters_summary.pdf'
plt.savefig(summary_pdf, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'参数汇总PDF已保存至: {summary_pdf}')

plt.show()
print('两张图片绘制完成！')
