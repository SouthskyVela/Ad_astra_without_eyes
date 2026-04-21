# 造父变星光变曲线音频化技术报告

## Technical Report: Cepheid Variable Star Light Curve Sonification

**项目**: Ad astra without eyes - Variable Star Sonification  
**对象**: Delta Cephei (造父变星)  
**数据**: 最近12个周期的光变曲线观测数据  
**日期**: 2026-04-21  
**版本**: v2.0

---

## 目录

1. [项目概述](#1-项目概述)
2. [数学原理](#2-数学原理)
3. [方案设计](#3-方案设计)
4. [拟合方法对比](#4-拟合方法对比)
5. [实现细节](#5-实现细节)
6. [结果分析](#6-结果分析)
7. [结论与展望](#7-结论与展望)

---

## 1. 项目概述

### 1.1 背景

造父变星(Cepheid Variable Stars)是一类周期性脉动的恒星，其光度随时间周期性变化。这种周期性变化是宇宙距离测量的重要标准烛光。本项目旨在通过**可听化(Sonification)**技术，将造父变星的光变曲线数据转换为音频信号，为视障人士和天文爱好者提供一种全新的感知天体物理现象的方式。

### 1.2 数据说明

- **数据来源**: AAVSO (美国变星观测者协会)
- **数据格式**: JD (儒略日), Magnitude (星等), Uncertainty (不确定度), Band (波段), Observer Code (观测者代码)
- **观测点数**: 61个有效观测点
- **时间跨度**: 63.87天（约12个周期）
- **星等范围**: 3.1 - 4.3等

### 1.3 目标

开发多种音频化处理方案：
- **方案1**: 离散数据 → 离散音高 → DFT频谱
- **方案2**: 傅里叶级数拟合 → 连续音高 → FFT频谱
- **方案3**: 多项式拟合 → 连续音高 → FFT频谱

---

## 2. 数学原理

### 2.1 光变曲线建模

#### 2.1.1 离散表示

光变曲线作为时间序列数据，离散表示为：

$$
\{(t_i, m_i)\}_{i=0}^{N-1}
$$

其中 $t_i$ 为观测时间，$m_i$ 为对应星等，$N$ 为观测点数。

#### 2.1.2 连续表示

##### (1) 三次样条插值 (Cubic Spline)

$$
S(t) = \sum_{i=0}^{N-2} c_i B_i(t), \quad t \in [t_i, t_{i+1}]
$$

其中 $B_i(t)$ 为三次B样条基函数，满足插值条件和光滑条件。

##### (2) 多项式拟合 (Polynomial Fitting)

使用最小二乘法拟合 $d$ 阶多项式：

$$
m(t) = \sum_{k=0}^{d} a_k \cdot \left(\frac{t-t_0}{t_{N-1}-t_0}\right)^k
$$

**最优阶数选择**: 通过肘部法则(Elobow Method)和RMSE阈值确定，本项目中推荐 **15阶**。

### 2.2 星等到频率的映射

#### 2.2.1 映射函数

星等(magnitude)与亮度成对数关系，频率映射采用线性反比关系：

$$
f(m) = f_{\max} - \frac{m - m_{\min}}{m_{\max} - m_{\min}} \cdot (f_{\max} - f_{\min})
$$

其中：
- $m_{\min}, m_{\max}$: 观测星等的最小值和最大值
- $f_{\min} = 200$ Hz, $f_{\max} = 800$ Hz (多项式拟合方案)

**物理意义**: 星等越小（恒星越亮）→ 频率越高

### 2.3 傅里叶级数拟合

对于周期性光变曲线，使用傅里叶级数展开：

$$
m(t) = \frac{a_0}{2} + \sum_{n=1}^{N_h} \left[a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right)\right]
$$

其中：
- $P$: 光变周期（通过DFT主峰估计）
- $N_h$: 谐波次数（通常取20）
- 系数通过最小二乘法估计

### 2.4 离散傅里叶变换 (DFT) 与快速傅里叶变换 (FFT)

#### 2.4.1 DFT定义

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j \frac{2\pi}{N} kn}
$$

#### 2.4.2 窗函数

为减少频谱泄漏，使用汉宁窗(Hanning Window)：

$$
w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right)
$$

### 2.5 Chirp信号生成

Chirp信号是频率随时间变化的正弦波：

$$
s(t) = \sin\left(2\pi \int_0^t f(\tau) d\tau\right)
$$

离散实现通过相位累加：

$$
\phi[n] = 2\pi \sum_{i=0}^{n} \frac{f[i]}{f_s}, \quad s[n] = \sin(\phi[n])
$$

---

## 3. 方案设计

### 3.1 方案1: 离散处理 (Scheme 1: Discrete)

#### 3.1.1 时域音频化

**算法流程**:
1. 每个观测点 $(t_i, m_i)$ 映射到频率 $f_i = f(m_i)$
2. 计算时间间隔 $\Delta t_i = t_{i+1} - t_i$
3. 生成正弦波片段: $s_i(t) = \sin(2\pi f_i t)$
4. 拼接所有片段

**输出**: `scheme1_discrete_time.wav`

#### 3.1.2 频域音频化

**算法流程**:
1. 对61个原始数据点做DFT
2. 提取幅度 $|X[k]|$ 和相位 $\angle X[k]$
3. 从低频到高频依次播放频谱分量
4. 音高从200Hz扫描到2000Hz，响度表征频谱幅度

**输出**: `scheme1_discrete_freq.wav`

### 3.2 方案2: 傅里叶级数拟合 (Scheme 2: Fourier Series)

#### 3.2.1 时域音频化

**算法流程**:
1. 估计周期 $P$（通过DFT主峰）
2. 计算傅里叶系数 $a_0, a_n, b_n$
3. 生成高密度拟合数据（10000点）
4. 映射到频率并生成Chirp信号

**特点**: 强制周期性，适合严格周期变星

#### 3.2.2 频域音频化

**算法流程**:
1. 傅里叶级数拟合生成高密度数据（16384点）
2. 加窗FFT
3. 频谱扫描播放

### 3.3 方案3: 多项式拟合 (Scheme 3: Polynomial)

#### 3.3.1 核心思想

使用**多项式拟合**代替周期性假设，更适合非严格周期性或存在长期趋势的光变曲线。

#### 3.3.2 最优阶数选择

| 阶数 | RMSE | 评价 |
|------|------|------|
| 5 | 0.237 | 过于平滑，欠拟合 |
| 10 | 0.228 | 基本趋势 |
| **15** | **0.202** | **推荐：平衡精度与稳定性** |
| 20 | 0.169 | 边界振荡，过拟合 |
| 25 | 0.161 | 严重过拟合 |

**选择策略**: 肘部法则 + 8-15阶限制

#### 3.3.3 时域音频化

**输出**: `polynomial_time_deg15.wav`

**参数**:
- 音频时长: 8秒
- 频率范围: 200-800 Hz
- 星等映射: 线性反比

#### 3.3.4 频域音频化

**输出**: `polynomial_freq_deg15.wav`

**算法流程**:
1. 15阶多项式拟合
2. 8192点FFT
3. 取前500个低频分量
4. 从200Hz扫描到800Hz，每频点16ms
5. 响度与频谱幅度成正比

---

## 4. 拟合方法对比

### 4.1 方法比较

| 方法 | 周期假设 | 边界行为 | RMSE | 适用场景 |
|------|----------|----------|------|----------|
| 三次样条 | 无 | 自然外推 | ~0 | 精确插值 |
| 线性插值 | 无 | 线性 | ~0 | 快速简单 |
| 傅里叶级数 | 强制周期 | 周期连接 | 较高 | 严格周期变星 |
| **多项式拟合** | **无** | **需限制范围** | **中等** | **通用场景** |

### 4.2 多项式拟合优势

1. **无需周期假设**: 适合非严格周期性光变曲线
2. **平滑连续**: 高阶多项式可捕捉复杂趋势
3. **计算高效**: 最小二乘法求解，复杂度 $O(Nd^2)$
4. **可调精度**: 通过阶数控制拟合精度

### 4.3 过拟合防护

高阶多项式(>15)易出现边界振荡，采取以下措施：
- 阶数限制在8-15范围
- 预测值裁剪: $m_{\text{clip}} = \text{clip}(m, m_{\min}-0.5, m_{\max}+0.5)$
- 频率限制: $f = \text{clip}(f, f_{\min}, f_{\max})$

---

## 5. 实现细节

### 5.1 核心代码结构

```
listen_02_variableStar/
├── variable_star_sonification_v2.py    # 方案1&2: 离散/傅里叶级数
├── polynomial_fitting_sonification.py  # 方案3: 多项式拟合
├── fitting_comparison.py               # 多种拟合方法对比
└── TECHNICAL_REPORT.md                 # 本报告
```

### 5.2 关键代码片段

#### 5.2.1 多项式拟合

```python
def polynomial_fit(self, time, mag, degree):
    # 时间归一化到[0,1]，提高数值稳定性
    t_normalized = (time - time[0]) / (time[-1] - time[0])
    
    # 最小二乘拟合
    coeffs = np.polyfit(t_normalized, mag, degree)
    poly = np.poly1d(coeffs)
    
    # 预测函数
    def predict(t):
        t_norm = (t - time[0]) / (time[-1] - time[0])
        return poly(t_norm)
    
    return coeffs, predict, rmse
```

#### 5.2.2 频域音频生成

```python
def generate_freq_domain_audio(self, time, mag, poly_func, output_path):
    # 1. 高密度FFT
    N_fft = 8192
    t_fine = np.linspace(time[0], time[-1], N_fft)
    mag_fitted = poly_func(t_fine)
    
    # 2. 加窗FFT
    dft = fft.fft((mag_fitted - np.mean(mag_fitted)) * np.hanning(N_fft))
    
    # 3. 取前500个低频分量
    spectrum_mags = np.abs(dft[:500])
    
    # 4. 从低频到高频扫描
    for i in range(500):
        progress = i / 500
        audio_freq = FREQ_MIN + progress * (FREQ_MAX - FREQ_MIN)
        amplitude = 0.1 + 0.9 * spectrum_mags[i] / max_spectrum
        # 生成16ms纯音
```

### 5.3 中文字体设置

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
```

---

## 6. 结果分析

### 6.1 输出文件清单

| 文件 | 方案 | 说明 |
|------|------|------|
| `scheme1_discrete_time.wav` | 方案1 | 61个离散点 → 离散音高 |
| `scheme1_discrete_freq.wav` | 方案1 | 原始数据DFT频谱扫描 |
| `polynomial_time_deg15.wav` | 方案3 | 多项式拟合 → 连续音高 |
| `polynomial_freq_deg15.wav` | 方案3 | 多项式FFT频谱扫描 |

### 6.2 频谱分析结果

#### 6.2.1 主周期识别

从频谱分析识别造父变星主要周期：

$$
P_{\text{Cepheid}} \approx 5.37 \text{ days}
$$

对应频率：

$$
f_{\text{main}} = \frac{1}{P} \approx 0.186 \text{ cycles/day}
$$

#### 6.2.2 多项式拟合质量

- **阶数**: 15
- **RMSE**: 0.202
- **最大误差**: 0.538星等
- **残差标准差**: 0.202

### 6.3 音频特征对比

| 音频 | 时长 | 频率范围 | 特点 |
|------|------|----------|------|
| scheme1_discrete_time | 30s | 200-2000 Hz | 阶梯式频率变化 |
| scheme1_discrete_freq | 30s | 200-2000 Hz | 61个频点扫描 |
| polynomial_time_deg15 | 8s | 200-800 Hz | 平滑连续变化 |
| polynomial_freq_deg15 | 8s | 200-800 Hz | 500个频点扫描 |

---

## 7. 结论与展望

### 7.1 技术总结

本项目实现了三种造父变星光变曲线音频化处理方案：

1. **方案1（离散处理）**:
   - 直接处理原始61个观测点
   - 优点: 保留数据原始特性
   - 缺点: 频率跳变明显

2. **方案2（傅里叶级数）**:
   - 强制周期性拟合
   - 优点: 物理意义明确
   - 缺点: 对非周期趋势敏感

3. **方案3（多项式拟合）**:
   - **无需周期假设，推荐用于通用场景**
   - 优点: 平滑连续，精度可调
   - 缺点: 高阶需防过拟合

### 7.2 关键技术点

| 技术 | 应用 |
|------|------|
| 多项式拟合 | 无周期假设的曲线拟合 |
| 肘部法则 | 最优阶数自动选择 |
| 边界裁剪 | 防止过拟合导致的无效值 |
| 频谱扫描 | 从低频到高频依次播放 |
| 幅度映射 | 响度表征频谱分量大小 |

### 7.3 应用价值

- **科学教育**: 为天文学教学提供多感官体验
- **无障碍天文**: 帮助视障人士"聆听"天体物理现象
- **数据分析**: 通过听觉发现周期性特征

### 7.4 未来工作

1. **立体声扩展**: 将时间映射到立体声声像
2. **多星对比**: 同时处理多颗变星
3. **触觉反馈**: 结合温度/振动设备
4. **深度学习**: 使用神经网络优化音频映射

---

## 附录

### A. 程序文件说明

| 程序 | 功能 |
|------|------|
| `variable_star_sonification_v2.py` | 方案1&2实现（离散+傅里叶级数） |
| `polynomial_fitting_sonification.py` | 方案3实现（多项式拟合） |
| `fitting_comparison.py` | 5种拟合方法对比分析 |

### B. 主要参数配置

```python
# 多项式拟合参数
OPTIMAL_DEGREE = 15
FREQ_MIN = 200      # Hz
FREQ_MAX = 800      # Hz
AUDIO_DURATION = 8  # seconds
SAMPLE_RATE = 44100 # Hz

# FFT参数
N_FFT = 8192
N_SPECTRUM_POINTS = 500
```

### C. 参考文献

1. AAVSO Data File Format: https://www.aavso.org/format-data-file
2. NumPy Polyfit Documentation
3. SciPy FFT Documentation
4. ISO 226:2023 - Normal Equal-Loudness-Level Contours

---

**报告编制**: AI Assistant  
**版本**: v2.0  
**更新日期**: 2026-04-21
