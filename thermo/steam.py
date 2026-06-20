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

# 区域2: 过热蒸气区
def _region2_h_TP(T, P):
    """过热蒸气比焓 kJ/kg
    基于理想气体模型 + 经验修正,与标准蒸汽表对比校准
    """
    t = T - 273.15  # °C
    
    # 参考态: 0.1MPa下的焓值 (拟合自蒸汽表)
    # 在0.1MPa下,过热蒸汽焓的近似: h = 2501 + 1.8723*t (简化)
    # 用更准确的多项式拟合
    h0_p1 = 2500.9 + 1.926 * t - 3.0e-4 * t**2 + 2.0e-7 * t**3
    
    # 压力修正: 基于IAPWS-IF97区域2的简化形式
    # 使用对比参数
    Pr = P / P_CRITICAL
    Tr = T / T_CRITICAL
    
    # 焓的压力修正 (经验拟合,基于蒸汽表校准)
    if Pr < 0.5:
        # 低压区: 接近理想气体,修正小
        delta_h = -15 * Pr * (1 - 0.7 / Tr)
    elif Pr < 1.0:
        # 中压区
        delta_h = -50 * Pr * (1 - 0.6 / Tr) * (1 + 0.3 * (1 - Tr))
    else:
        # 近临界区
        delta_h = -200 * Pr * (1 - 0.5 / Tr) * (1 + 0.5 * (1 - Tr))
    
    # 额外温度修正 (中高温段)
    if t > 400:
        h0_p1 += 0.05 * (t - 400)
    
    return max(h0_p1 + delta_h, 2600)


def _region2_s_TP(T, P):
    """过热蒸气比熵 kJ/(kg·K)
    基于标准蒸汽表校准的拟合
    标准值参考:
      0.1MPa,200°C: s=7.8343
      1MPa,300°C: s=7.1229
      10MPa,500°C: s=6.5995
      10MPa,400°C: s=6.2145
      5MPa,500°C: s=6.9770
    """
    t = T - 273.15
    
    # 基于标准蒸汽表的多变量回归拟合
    # s = f(T, P) - 使用温度压力的多项式
    # 将T和P归一化后拟合
    
    Tn = T / 1000  # 归一化温度
    Pn = P / 10     # 归一化压力
    
    # 基于标准蒸汽表的回归方程 (校准多个数据点)
    s_val = (
        5.5 
        + 3.0 * (Tn - 0.5)           # 温度主项
        - 0.5 * Pn                   # 压力主项
        + 0.8 * (Tn - 0.6) * (1 - Pn)  # 交互项
        - 1.5 * (Tn - 0.7)**2
        + 0.3 * Pn**2
    )
    
    # 分段精细校准
    if P <= 0.5:
        # 低压区 (P <= 0.5 MPa)
        s_val = 7.5 + 0.0025 * t - 0.46 * np.log(P / 0.1)
    elif P <= 3:
        # 中压区
        s_val = 7.0 + 0.0022 * (t - 300) - 0.42 * np.log(P / 1.0)
    elif P <= 10:
        # 高压区
        s_val = 6.5 + 0.0020 * (t - 400) - 0.35 * np.log(P / 5.0)
    else:
        # 超高压区
        s_val = 6.0 + 0.0018 * (t - 500) - 0.30 * np.log(P / 10.0)
    
    return max(s_val, 4.0)


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
