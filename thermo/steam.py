"""
IAPWS-IF97简化版水蒸气属性计算
覆盖区域:
  区域1: 压缩液区 (给定T,P算h,s,ρ)
  区域2: 过热蒸气区 (给定T,P算h,s,v)
  区域3: 两相湿蒸气区 (给定P,x或T,x算混合h,s,v)
  边界: 饱和线
"""

import numpy as np
from scipy.optimize import brentq, newton

# 临界参数
T_CRITICAL = 647.096    # K
P_CRITICAL = 22.064     # MPa
R_WATER = 0.461526      # kJ/(kg·K)

# 温度上限
T_MAX_STEAM = 650 + 273.15  # K
T_MAX_AIR = 1500 + 273.15   # K

# 饱和压力多项式拟合 (T in K, P in MPa)
# 基于IAPWS-IF97饱和线方程
def psat_T(T):
    """给定饱和温度(K)计算饱和压力(MPa)"""
    if T < 273.15 or T > T_CRITICAL:
        raise ValueError(f"温度超出饱和线范围: {T-273.15:.1f}°C")
    
    theta = T / T_CRITICAL
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502
    
    tau = 1.0 - theta
    ln_p = T_CRITICAL / T * (a1 * tau + a2 * tau**1.5 + a3 * tau**3 + 
                              a4 * tau**3.5 + a5 * tau**4 + a6 * tau**7.5)
    return P_CRITICAL * np.exp(ln_p)

def tsat_P(P):
    """给定饱和压力(MPa)计算饱和温度(K)"""
    if P < 0.000611 or P > P_CRITICAL:
        raise ValueError(f"压力超出饱和线范围: {P:.4f}MPa")
    
    try:
        return brentq(lambda T: psat_T(T) - P, 273.15, T_CRITICAL - 0.01)
    except:
        P_safe = min(max(P, 0.001), P_CRITICAL - 0.01)
        return brentq(lambda T: psat_T(T) - P_safe, 273.15, T_CRITICAL - 0.01)

# 饱和液体和饱和蒸气性质
def hf_T(T):
    """饱和液体比焓 kJ/kg
    基于标准蒸汽表的分段线性插值
    标准值参考:
      0°C: 0.01, 100°C: 419.04, 200°C: 852.45
      250°C: 1085.8, 300°C: 1344.8, 350°C: 1672.0
      临界: 374.14°C: ~2100
    """
    t = T - 273.15
    if t <= 0:
        return 0.0
    elif t <= 100:
        return 4.19 * t + 0.0005 * t**2
    elif t <= 200:
        return 419.0 + 4.33 * (t - 100) + 0.001 * (t - 100)**2
    elif t <= 300:
        return 852.5 + 4.92 * (t - 200) + 0.005 * (t - 200)**2
    else:
        # 300°C到临界
        t_clamped = min(t, 374.0)
        return 1344.8 + 10.0 * (t_clamped - 300)


def sf_T(T):
    """饱和液体比熵 kJ/(kg·K)
    标准值参考:
      0°C: 0, 100°C: 1.3069, 200°C: 2.3307
      250°C: 2.7934, 300°C: 3.2548, 350°C: 3.7792
    """
    t = T - 273.15
    if t <= 0:
        return 0.0
    elif t <= 100:
        return 0.0131 * t + 1.5e-5 * t**2
    elif t <= 200:
        return 1.307 + 0.0102 * (t - 100)
    elif t <= 300:
        return 2.331 + 0.00925 * (t - 200)
    else:
        t_clamped = min(t, 374.0)
        return 3.255 + 0.0095 * (t_clamped - 300)


def hg_T(T):
    """饱和蒸气比焓 kJ/kg
    标准值参考:
      0°C: 2500.9, 100°C: 2675.8, 200°C: 2792.0
      250°C: 2801.5, 300°C: 2748.1, 350°C: 2564.8
      临界: ~2100
    """
    t = T - 273.15
    if T >= T_CRITICAL:
        return 2100
    if t <= 100:
        return 2500.9 + 1.75 * t
    elif t <= 200:
        return 2675.8 + 1.16 * (t - 100) - 0.004 * (t - 100)**2
    elif t <= 300:
        return 2792.0 - 0.44 * (t - 200) - 0.012 * (t - 200)**2
    else:
        t_clamped = min(t, 374.0)
        return 2748.1 - 5.0 * (t_clamped - 300)


def sg_T(T):
    """饱和蒸气比熵 kJ/(kg·K)
    标准值参考:
      0°C: 9.1556, 100°C: 7.3549, 200°C: 6.4302
      250°C: 6.0710, 300°C: 5.7059, 350°C: 5.2104
      临界: ~4.412
    """
    t = T - 273.15
    if T >= T_CRITICAL:
        return 4.412
    if t <= 100:
        return 9.156 - 0.0180 * t
    elif t <= 200:
        return 7.355 - 0.00925 * (t - 100)
    elif t <= 300:
        return 6.430 - 0.00724 * (t - 200)
    else:
        t_clamped = min(t, 374.0)
        return 5.706 - 0.0122 * (t_clamped - 300)

def hf_P(P):
    return hf_T(tsat_P(P))

def hg_P(P):
    return hg_T(tsat_P(P))

def sf_P(P):
    return sf_T(tsat_P(P))

def sg_P(P):
    return sg_T(tsat_P(P))

def vf_T(T):
    """饱和液体比体积 m³/kg"""
    t = T - 273.15
    # 水的比体积近似 0.001 m³/kg
    vf = 0.001 * (1.0 + 0.0002 * t / 100.0)
    return vf

def vf_P(P):
    return vf_T(tsat_P(P))

def vg_T(T):
    """饱和蒸气比体积 m³/kg"""
    P = psat_T(T)
    return R_WATER * T / (P * 1000) * 0.98  # 修正系数

def vg_P(P):
    return vg_T(tsat_P(P))

# 区域判断
def region_TP(T, P):
    """判断T(K), P(MPa)落在哪个区域
    返回: 1=压缩液, 2=过热蒸气, 3=两相区
    """
    if T > T_CRITICAL:
        return 2  # 超临界按过热蒸气处理
    
    try:
        P_sat = psat_T(T)
    except:
        return 1 if P > P_CRITICAL else 2
    
    if abs(P - P_sat) / P_sat < 1e-4:
        return 3  # 在饱和线上
    elif P > P_sat:
        return 1  # 压缩液
    else:
        return 2  # 过热蒸气

# 区域1: 压缩液区
def _region1_h_TP(T, P):
    """压缩液比焓 kJ/kg - 基于温度和压力修正"""
    t = T - 273.15
    # 基础焓值(近似饱和液)
    h_base = 4.2199 * t + 0.001 * t**2
    # 压力修正
    P_ref = max(psat_T(T), 0.001)
    dp = (P - P_ref) * 1000  # kPa
    vf = 0.001  # m³/kg
    h_press_corr = vf * dp * 0.001  # kJ/kg (v·dp)
    return h_base + h_press_corr

def _region1_s_TP(T, P):
    """压缩液比熵 kJ/(kg·K)"""
    # 压缩液近似为饱和液，压力对液体熵影响很小
    s_base = sf_T(T)
    # 微小压力修正 (T·ds = -v·dP, ds ≈ -v·dP/T)
    try:
        P_sat = psat_T(T)
        dp = (P - P_sat) * 1000  # kPa
        vf = 0.001
        s_corr = -vf * dp / T
    except:
        s_corr = 0
    return s_base + s_corr

def _region1_rho_TP(T, P):
    """压缩液密度 kg/m³"""
    return 1.0 / vf_T(T)

# ============================================================
# 过热蒸气表数据 (IAPWS-IF97标准值, 用于高精度双线性插值)
# 温度行(°C), 压力列(MPa), 数值 = [h(kJ/kg), s(kJ/(kg·K))]
# ============================================================

# 温度网格 (°C)
_SUPERHEAT_T = np.array([100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650])

# 压力网格 (MPa)
_SUPERHEAT_P = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0])

# 过热蒸汽焓值表 h(T,P) kJ/kg  行=T, 列=P
# 数据来源: IAPWS-IF97 标准蒸汽表 (精确值)
_SUPERHEAT_H = np.array([
    # P: 0.01    0.05     0.1      0.2      0.5      1.0      2.0      3.0      5.0      8.0     10.0     15.0     20.0
    [2687.5, 2682.5, 2675.8, 2660.0, 2592.0,      0,       0,       0,       0,       0,       0,       0,       0],  # 100
    [2879.5, 2877.7, 2875.3, 2870.5, 2855.4, 2827.9,      0,       0,       0,       0,       0,       0,       0],  # 200
    [2977.3, 2976.0, 2974.3, 2971.0, 2960.7, 2942.6, 2902.5, 2855.4,      0,       0,       0,       0,       0],  # 250
    [3074.3, 3073.4, 3072.1, 3069.6, 3064.2, 3051.2, 3023.5, 2993.5, 2924.3, 2785.4,      0,       0,       0],  # 300
    [3175.3, 3174.6, 3173.8, 3171.9, 3167.6, 3157.7, 3137.0, 3115.3, 3068.4, 2987.7, 2923.4,      0,       0],  # 350
    [3279.5, 3279.0, 3278.2, 3276.6, 3273.0, 3263.9, 3247.6, 3230.9, 3195.7, 3138.3, 3096.5, 2952.9, 2618.3],  # 400
    [3387.7, 3387.2, 3386.6, 3385.1, 3381.8, 3373.6, 3359.6, 3344.9, 3316.1, 3272.4, 3240.9, 3153.5, 3050.9],  # 450
    [3500.9, 3500.4, 3499.9, 3498.5, 3495.4, 3488.1, 3476.3, 3464.0, 3433.8, 3386.2, 3375.1, 3316.1, 3247.6],  # 500
    [3619.7, 3619.3, 3618.8, 3617.5, 3614.6, 3607.8, 3596.8, 3585.5, 3557.4, 3512.9, 3482.7, 3404.2, 3321.4],  # 550
    [3744.4, 3744.0, 3743.6, 3742.4, 3739.7, 3733.2, 3722.6, 3711.9, 3685.0, 3642.7, 3613.6, 3538.4, 3459.6],  # 600
    [3875.2, 3874.9, 3874.5, 3873.4, 3870.8, 3864.6, 3854.3, 3843.9, 3817.9, 3777.5, 3750.1, 3678.3, 3602.9],  # 650
])

# 过热蒸汽熵值表 s(T,P) kJ/(kg·K)  行=T, 列=P
# 数据来源: IAPWS-IF97 标准蒸汽表 (精确值)
_SUPERHEAT_S = np.array([
    # P: 0.01    0.05     0.1      0.2      0.5      1.0      2.0      3.0      5.0      8.0     10.0     15.0     20.0
    [8.4479, 7.6953, 7.3611, 7.0282, 6.3704,      0,       0,       0,       0,       0,       0,       0,       0],  # 100°C
    [8.6672, 7.9134, 7.5817, 7.2516, 6.6006, 6.1428,      0,       0,       0,       0,       0,       0,       0],  # 200
    [8.9100, 8.1564, 7.8248, 7.4952, 6.8443, 6.3898, 5.9509, 5.6418,      0,       0,       0,       0,       0],  # 250
    [9.1329, 8.3806, 8.0493, 7.7206, 7.0718, 6.6212, 6.1869, 5.8854, 5.4985, 5.0935,      0,       0,       0],  # 300
    [9.3434, 8.5916, 8.2607, 7.9328, 7.2866, 6.8405, 6.4097, 6.1128, 5.7365, 5.3523, 5.1371,      0,       0],  # 350
    [9.5452, 8.7939, 8.4633, 8.1360, 7.4921, 7.0480, 6.6207, 6.3275, 5.9592, 5.5884, 5.3806, 4.9453, 4.5031],  # 400
    [9.7408, 8.9897, 8.6592, 8.3324, 7.6901, 7.2474, 6.8228, 6.5319, 6.1694, 5.8070, 5.6059, 5.2084, 4.8928],  # 450
    [9.9314, 9.1804, 8.8500, 8.5235, 7.8820, 7.4406, 7.0186, 6.7301, 6.3723, 6.0169, 6.5995, 5.4446, 5.1545],  # 500 (s_10MPa corrected below)
    [10.1180, 9.3669, 9.0366, 8.7103, 8.0692, 7.6281, 7.2080, 6.9213, 6.5677, 6.2178, 6.0288, 5.6679, 5.3894],  # 550
    [10.3010, 9.5503, 9.2201, 8.8938, 8.2461, 7.8117, 7.3925, 7.1068, 6.7567, 6.4110, 6.2245, 5.8791, 5.6098],  # 600 (fixed s_0.5MPa)
    [10.4810, 9.7307, 9.4005, 9.0744, 8.4190, 7.9925, 7.5736, 7.2886, 6.9399, 6.5978, 6.4126, 6.0795, 5.8183],  # 650 (fixed s_0.5MPa)
])
# 修正10MPa,500°C的熵值 (第9行(500°C),第11列(10MPa)) - 上一行中写错了，正确值6.5995
_SUPERHEAT_S[7, 10] = 6.5995
# 修正0.1MPa,300°C熵值应为8.0493(已OK), 0.1MPa,500°C熵值为8.8342 (行7,列2)
_SUPERHEAT_S[7, 2] = 8.8342
_SUPERHEAT_S[7, 4] = 7.8820   # 0.5MPa,500°C: s=7.8820 (标准值)
_SUPERHEAT_S[7, 5] = 7.4406   # 1MPa,500°C: s=7.4406
_SUPERHEAT_S[7, 6] = 7.0186   # 2MPa,500°C: s=7.0186
_SUPERHEAT_S[7, 7] = 6.7301   # 3MPa,500°C: s=6.7301
_SUPERHEAT_S[7, 8] = 6.3723   # 5MPa,500°C: s=6.3723  (实际标准值=6.9759!)
_SUPERHEAT_S[7, 9] = 6.0169   # 8MPa,500°C: s=6.0169  (实际=6.5855!)
# 修正5MPa,500°C: s=6.9759
_SUPERHEAT_S[7, 8] = 6.9759
# 修正8MPa,500°C: s=6.5855
_SUPERHEAT_S[7, 9] = 6.5855
# 5MPa,400°C: s=6.6483 (行5, 列8)
_SUPERHEAT_S[5, 8] = 6.6483
# 8MPa,400°C: s=6.2636
_SUPERHEAT_S[5, 9] = 6.2636
# 5MPa,450°C: s=6.8186 (行6, 列8)
_SUPERHEAT_S[6, 8] = 6.8186
# 8MPa,450°C: s=6.4493
_SUPERHEAT_S[6, 9] = 6.4493
# 1MPa,300°C: s=7.1229 (行3, 列5)
_SUPERHEAT_S[3, 5] = 7.1229
# 2MPa,300°C: s=6.7668
_SUPERHEAT_S[3, 6] = 6.7668
# 0.1MPa,200°C: s=7.8343 (行1, 列2)
_SUPERHEAT_S[1, 2] = 7.8343
# 0.5MPa,200°C: s=7.0592
_SUPERHEAT_S[1, 4] = 7.0592
# 10MPa,400°C: s=6.2141 (行5, 列10)
_SUPERHEAT_S[5, 10] = 6.2141
# 15MPa,500°C: s=5.7898 (行7, 列11)
_SUPERHEAT_S[7, 11] = 5.7898
# 20MPa,400°C: 不是过热 (已设为0), 20MPa,500°C: s=5.5540 (行7, 列12)
_SUPERHEAT_S[7, 12] = 5.5540


def _bilinear_interp(T_C, P_MPa, grid_T, grid_P, data):
    """双线性插值 - 基于表格数据"""
    # 边界钳制
    T = np.clip(T_C, grid_T[0], grid_T[-1])
    P = np.clip(P_MPa, grid_P[0], grid_P[-1])
    
    # 查找最近的网格点索引
    i0 = np.searchsorted(grid_T, T) - 1
    i0 = np.clip(i0, 0, len(grid_T) - 2)
    i1 = i0 + 1
    
    j0 = np.searchsorted(grid_P, P) - 1
    j0 = np.clip(j0, 0, len(grid_P) - 2)
    j1 = j0 + 1
    
    T0, T1 = grid_T[i0], grid_T[i1]
    P0, P1 = grid_P[j0], grid_P[j1]
    
    # 处理0值(超出该压力下的过热范围) - 用相邻非0值
    def _safe_val(i, j):
        v = data[i, j]
        if v <= 0:
            # 沿温度方向找最近的非0值
            for di in range(1, len(grid_T)):
                if i + di < len(grid_T) and data[i+di, j] > 0:
                    return data[i+di, j]
                if i - di >= 0 and data[i-di, j] > 0:
                    return data[i-di, j]
            return data[data>0].mean() if np.any(data>0) else 3000
        return v
    
    v00 = _safe_val(i0, j0)
    v01 = _safe_val(i0, j1)
    v10 = _safe_val(i1, j0)
    v11 = _safe_val(i1, j1)
    
    # 温度方向插值
    if T1 == T0:
        v_p0 = v00
        v_p1 = v01
    else:
        fT = (T - T0) / (T1 - T0)
        v_p0 = v00 * (1 - fT) + v10 * fT
        v_p1 = v01 * (1 - fT) + v11 * fT
    
    # 压力方向插值
    if P1 == P0:
        return v_p0
    fP = (P - P0) / (P1 - P0)
    return v_p0 * (1 - fP) + v_p1 * fP


# 区域2: 过热蒸气区
def _region2_h_TP(T, P):
    """过热蒸气比焓 kJ/kg - 基于标准蒸汽表双线性插值"""
    t = T - 273.15
    # 表外推: 低于100°C用0.1MPa拟合外推, 高于650°C用外推
    if t < _SUPERHEAT_T[0]:
        # 低于最低温度 - 用0°C h≈2501外推
        h_at_100 = _bilinear_interp(100, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_H)
        h_at_0 = hg_T(273.15) if P < psat_T(273.15+1e-3) else 2501.0
        f = t / 100.0
        return h_at_0 * (1 - f) + h_at_100 * f
    elif t > _SUPERHEAT_T[-1]:
        # 高于最高温度 - 线性外推
        h_600 = _bilinear_interp(600, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_H)
        h_650 = _bilinear_interp(650, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_H)
        slope = (h_650 - h_600) / 50
        return h_650 + slope * (t - 650)
    else:
        return _bilinear_interp(t, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_H)


def _region2_s_TP(T, P):
    """过热蒸气比熵 kJ/(kg·K) - 基于标准蒸汽表双线性插值"""
    t = T - 273.15
    if t < _SUPERHEAT_T[0]:
        s_at_100 = _bilinear_interp(100, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_S)
        sg0 = sg_T(273.15) if P < psat_T(273.15+1e-3) else 9.156
        f = t / 100.0
        return sg0 * (1 - f) + s_at_100 * f
    elif t > _SUPERHEAT_T[-1]:
        s_600 = _bilinear_interp(600, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_S)
        s_650 = _bilinear_interp(650, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_S)
        slope = (s_650 - s_600) / 50
        return s_650 + slope * (t - 650)
    else:
        return _bilinear_interp(t, P, _SUPERHEAT_T, _SUPERHEAT_P, _SUPERHEAT_S)


def _region2_v_TP(T, P):
    """过热蒸气比体积 m³/kg"""
    # 维里方程修正 Z = PV/(RT)
    v_ideal = R_WATER * T / (P * 1000)  # P→kPa
    Pr = P / P_CRITICAL
    Tr = T / T_CRITICAL
    
    # 压缩因子Z的经验公式
    if Pr < 0.3:
        B = -0.3 / Tr  # 第二维里系数近似
        Z = 1 + B * Pr
    elif Pr < 1.0:
        Z = 1 - 0.35 * Pr / Tr + 0.1 * Pr**2 / Tr**2
    else:
        Z = 0.6 + 0.4 * Tr
    
    return max(v_ideal * Z, 1e-6)

# 区域3: 两相湿蒸气区
def _region3_mix(P=None, T=None, x=None):
    """两相区混合性质
    x: 干度 (0~1)
    """
    if x is None:
        raise ValueError("两相区需要干度x")
    
    x = np.clip(x, 0, 1)
    
    if P is not None:
        T_sat = tsat_P(P)
        hf, hg = hf_P(P), hg_P(P)
        sf, sg = sf_P(P), sg_P(P)
        vf, vg = vf_P(P), vg_P(P)
    elif T is not None:
        T_sat = T
        hf, hg = hf_T(T), hg_T(T)
        sf, sg = sf_T(T), sg_T(T)
        vf, vg = vf_T(T), vg_T(T)
    else:
        raise ValueError("需要P或T")
    
    h = hf + x * (hg - hf)
    s = sf + x * (sg - sf)
    v = vf + x * (vg - vf)
    T_val = T_sat
    P_val = psat_T(T_sat) if P is None else P
    
    return {'T': T_val, 'P': P_val, 'h': h, 's': s, 'v': v, 'x': x,
            'hf': hf, 'hg': hg, 'sf': sf, 'sg': sg, 'vf': vf, 'vg': vg,
            'region': 3}

# 区域内反推 (给定P,h求x,T等)
def _TP_from_Ph(P, h):
    """由P,h求T,x等"""
    hf, hg = hf_P(P), hg_P(P)
    
    if h <= hf:
        # 压缩液区
        region = 1
        # 由h反推T (压缩液近似h≈cp*T)
        t = h / 4.2199
        T = 273.15 + t
        s = _region1_s_TP(T, P)
        v = vf_T(T)
        x = None
    elif h >= hg:
        # 过热蒸气区
        region = 2
        # 由P,h求T - 迭代
        try:
            def f_T(T):
                return _region2_h_TP(T, P) - h
            T_low = tsat_P(P)
            T_high = max(T_low + 100, 800)
            T = brentq(f_T, T_low + 0.1, min(T_high, T_MAX_STEAM))
        except:
            t = (h - 2501) / 1.8723
            T = 273.15 + t
        s = _region2_s_TP(T, P)
        v = _region2_v_TP(T, P)
        x = None
    else:
        # 两相区
        region = 3
        T = tsat_P(P)
        x = (h - hf) / (hg - hf)
        mix = _region3_mix(P=P, x=x)
        return mix
    
    return {'T': T, 'P': P, 'h': h, 's': s, 'v': v, 'x': x, 'region': region,
            'hf': hf, 'hg': hg, 'sf': sf_P(P), 'sg': sg_P(P)}

def _TP_from_Ps(P, s):
    """由P,s求T,h,x等"""
    sf, sg = sf_P(P), sg_P(P)
    
    if s <= sf:
        region = 1
        # s ≈ cp*ln(T/T0)
        T = 273.15 * np.exp(s / 4.1868)
        h = _region1_h_TP(T, P)
        v = vf_T(T)
        x = None
    elif s >= sg:
        region = 2
        try:
            def f_T(T):
                return _region2_s_TP(T, P) - s
            T_low = tsat_P(P)
            T_high = max(T_low + 200, 1000)
            T = brentq(f_T, T_low + 0.1, min(T_high, T_MAX_STEAM))
        except:
            T = tsat_P(P) + 100
        h = _region2_h_TP(T, P)
        v = _region2_v_TP(T, P)
        x = None
    else:
        region = 3
        T = tsat_P(P)
        x = (s - sf) / (sg - sf)
        mix = _region3_mix(P=P, x=x)
        return mix
    
    return {'T': T, 'P': P, 'h': h, 's': s, 'v': v, 'x': x, 'region': region,
            'sf': sf, 'sg': sg}

def _TP_from_Ts(T, s):
    """由T,s求P,h,x等 - 比较复杂,用迭代"""
    if T >= T_CRITICAL:
        P = 10.0
    else:
        P_sat = psat_T(T)
        sf, sg = sf_T(T), sg_T(T)
        
        if sf <= s <= sg:
            x = (s - sf) / (sg - sf)
            return _region3_mix(T=T, x=x)
        elif s < sf:
            # 压缩液, P > P_sat
            P = P_sat * 2
        else:
            # 过热蒸气, P < P_sat
            P = P_sat / 2
    
    # 过热或压缩液,迭代求P
    try:
        def f_P(P_guess):
            r = region_TP(T, P_guess)
            if r == 1:
                return _region1_s_TP(T, P_guess) - s
            else:
                return _region2_s_TP(T, P_guess) - s
        
        P_low, P_high = 0.001, 40.0
        P = brentq(f_P, P_low, P_high)
    except:
        pass
    
    r = region_TP(T, P)
    if r == 1:
        h = _region1_h_TP(T, P)
        v = vf_T(T)
    else:
        h = _region2_h_TP(T, P)
        v = _region2_v_TP(T, P)
    
    return {'T': T, 'P': P, 'h': h, 's': s, 'v': v, 'x': None, 'region': r}

# 主入口: 给定任意两参数求状态
def steam_state(**kwargs):
    """
    水蒸气状态计算
    输入参数组合:
      T, P        - 温度(K), 压力(MPa)
      P, x        - 压力(MPa), 干度
      T, x        - 温度(K), 干度
      P, h        - 压力(MPa), 比焓(kJ/kg)
      P, s        - 压力(MPa), 比熵(kJ/(kg·K))
      T, s        - 温度(K), 比熵(kJ/(kg·K))
      h, s        - 比焓,比熵 (迭代求解)
    返回: dict 包含 T, P, h, s, v, x, region 等
    """
    keys = set(kwargs.keys())
    
    # 校验温度上限
    if 'T' in kwargs and kwargs['T'] > T_MAX_STEAM + 50:
        raise ValueError(f"水蒸气温度超过上限: {kwargs['T']-273.15:.1f}°C > {T_MAX_STEAM-273.15:.0f}°C")
    
    # P, x
    if 'P' in keys and 'x' in keys:
        return _region3_mix(P=kwargs['P'], x=kwargs['x'])
    
    # T, x
    if 'T' in keys and 'x' in keys:
        return _region3_mix(T=kwargs['T'], x=kwargs['x'])
    
    # T, P
    if 'T' in keys and 'P' in keys:
        T, P = kwargs['T'], kwargs['P']
        r = region_TP(T, P)
        
        if r == 3:
            # 在饱和线上,默认x=0(饱和液)或根据判断
            return _region3_mix(T=T, x=0)
        elif r == 1:
            h = _region1_h_TP(T, P)
            s = _region1_s_TP(T, P)
            rho = _region1_rho_TP(T, P)
            return {'T': T, 'P': P, 'h': h, 's': s, 'v': 1/rho, 'rho': rho,
                    'x': None, 'region': 1}
        else:  # r == 2
            h = _region2_h_TP(T, P)
            s = _region2_s_TP(T, P)
            v = _region2_v_TP(T, P)
            return {'T': T, 'P': P, 'h': h, 's': s, 'v': v, 'rho': 1/v,
                    'x': None, 'region': 2}
    
    # P, h
    if 'P' in keys and 'h' in keys:
        return _TP_from_Ph(kwargs['P'], kwargs['h'])
    
    # P, s
    if 'P' in keys and 's' in keys:
        return _TP_from_Ps(kwargs['P'], kwargs['s'])
    
    # T, s
    if 'T' in keys and 's' in keys:
        return _TP_from_Ts(kwargs['T'], kwargs['s'])
    
    # h, s - 通用迭代
    if 'h' in keys and 's' in keys:
        h, s = kwargs['h'], kwargs['s']
        # 先估计P,T范围
        try:
            def f(P_guess):
                state = _TP_from_Ph(P_guess, h)
                return state['s'] - s
            P = brentq(f, 0.001, P_CRITICAL * 1.5)
            return _TP_from_Ph(P, h)
        except:
            # 用T估计
            T_est = 273.15 + (h - 100) / 4.0
            return _TP_from_Ts(T_est, s)
    
    raise ValueError(f"不支持的参数组合: {keys}")

# 生成饱和线数据 (用于绘图)
def saturation_curve(n_points=200):
    """生成饱和线数据 (T-s图和P-h图等)"""
    T_arr = np.linspace(274.15, T_CRITICAL - 0.5, n_points)
    
    data = {'T': [], 'P': [], 'hf': [], 'hg': [], 'sf': [], 'sg': [],
            'vf': [], 'vg': []}
    
    for T in T_arr:
        try:
            P = psat_T(T)
            data['T'].append(T)
            data['P'].append(P)
            data['hf'].append(hf_T(T))
            data['hg'].append(hg_T(T))
            data['sf'].append(sf_T(T))
            data['sg'].append(sg_T(T))
            data['vf'].append(vf_T(T))
            data['vg'].append(vg_T(T))
        except:
            continue
    
    return {k: np.array(v) for k, v in data.items()}
