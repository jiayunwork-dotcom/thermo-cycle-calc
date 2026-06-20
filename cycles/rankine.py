"""
Rankine循环计算
- 基础Rankine循环
- 再热Rankine循环
- 回热Rankine循环 (开式/闭式加热器)
"""

import numpy as np
from thermo.state import StatePoint
from thermo.steam import steam_state, tsat_P, psat_T, T_MAX_STEAM

MIN_TURBINE_QUALITY = 0.88  # 末级最低干度


class BaseCycle:
    """循环基类"""
    
    def __init__(self, name='Cycle'):
        self.name = name
        self.states = {}  # {label: StatePoint}
        self.processes = []  # [(label1, label2, process_type)]
        self.results = {}
        self.warnings = []
    
    def add_state(self, label, fluid='water', **kwargs):
        sp = StatePoint(fluid=fluid, label=label, **kwargs)
        self.states[label] = sp
        return sp
    
    def add_process(self, label1, label2, ptype=''):
        self.processes.append((label1, label2, ptype))
    
    def compute(self):
        raise NotImplementedError
    
    def efficiency(self):
        return self.results.get('eta', 0)
    
    def get_states_df(self):
        import pandas as pd
        data = []
        for label, sp in self.states.items():
            d = sp.to_dict()
            d['T_C'] = d['T'] - 273.15 if d['T'] else None
            data.append(d)
        return pd.DataFrame(data)


class RankineCycle(BaseCycle):
    """基础Rankine循环
    状态:
      1: 冷凝器出口 (饱和液, P_cond)
      2: 泵出口 (P_boiler)
      3: 锅炉出口 (过热蒸汽, T_boiler, P_boiler)
      4: 汽轮机出口 (P_cond)
    """
    
    def __init__(self, P_boiler=10, T_boiler=500+273.15, P_cond=0.01,
                 eta_pump=0.85, eta_turbine=0.90):
        """
        P_boiler: 锅炉压力 MPa
        T_boiler: 锅炉出口温度 K
        P_cond:   冷凝器压力 MPa
        eta_pump: 泵效率
        eta_turbine: 汽轮机等熵效率
        """
        super().__init__('基础Rankine循环')
        self.P_boiler = P_boiler
        self.T_boiler = T_boiler
        self.P_cond = P_cond
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        self.warnings = []
    
    def compute(self):
        P_b, T_b = self.P_boiler, self.T_boiler
        P_c = self.P_cond
        eta_p, eta_t = self.eta_pump, self.eta_turbine
        
        # 温度校验
        if T_b > T_MAX_STEAM:
            self.warnings.append(f"锅炉温度{T_b-273.15:.1f}°C超过上限{T_MAX_STEAM-273.15:.0f}°C")
        
        # 状态1: 冷凝器出口饱和液
        s1 = self.add_state('1', fluid='water', P=P_c, x=0)
        h1_val, s1_val, v1_val = s1.h, s1.s, s1.v
        
        # 状态2: 泵出口 (理想等熵)
        h2_is = h1_val + v1_val * (P_b - P_c) * 1000  # kJ/kg, v·dp
        h2_val = h1_val + (h2_is - h1_val) / eta_p
        s2_state = self.add_state('2', fluid='water', P=P_b, h=h2_val)
        
        # 状态3: 锅炉出口
        s3 = self.add_state('3', fluid='water', P=P_b, T=T_b)
        h3_val, s3_val = s3.h, s3.s
        
        # 状态4: 汽轮机出口 - 先等熵
        s4_is = steam_state(P=P_c, s=s3_val)
        h4_is = s4_is['h']
        x4_is = s4_is.get('x')
        h4_val = h3_val - eta_t * (h3_val - h4_is)
        s4 = self.add_state('4', fluid='water', P=P_c, h=h4_val)
        
        # 干度校验
        if s4.x is not None and s4.x < MIN_TURBINE_QUALITY:
            self.warnings.append(
                f"末级湿度过大: 出口干度x={s4.x:.4f} < {MIN_TURBINE_QUALITY}, 需调整参数"
            )
        
        # 能量计算
        w_pump = h2_val - h1_val          # 泵功 (消耗)
        w_turbine = h3_val - h4_val       # 汽轮机动 (输出)
        w_net = w_turbine - w_pump
        q_in = h3_val - s2_state.h        # 吸热量
        q_out = h4_val - h1_val           # 放热量
        
        eta = w_net / q_in if q_in > 0 else 0
        
        # Carnot效率 (同温限)
        T_h = T_b
        T_l = s1.T
        eta_carnot = 1 - T_l / T_h if T_h > 0 else 0
        
        # 第二定律 - 各组件熵产和㶲损失
        T0 = 298.15  # 环境温度
        s_gen_pump = s2_state.s - s1_val
        s_gen_turbine = s4.s - s3_val
        s_gen_boiler = s3_val - s2_state.s
        s_gen_condenser = s1_val - s4.s + q_out / T0
        
        exergy_destruction = {
            '泵': T0 * s_gen_pump,
            '锅炉': T0 * s_gen_boiler,
            '汽轮机': T0 * s_gen_turbine,
            '冷凝器': T0 * s_gen_condenser,
        }
        
        self.states['1'] = s1
        self.states['2'] = s2_state
        self.states['3'] = s3
        self.states['4'] = s4
        
        self.processes = [
            ('1', '2', '泵加压'),
            ('2', '3', '锅炉等压加热'),
            ('3', '4', '汽轮机膨胀'),
            ('4', '1', '冷凝器等压放热'),
        ]
        
        self.results = {
            'w_pump': w_pump,
            'w_turbine': w_turbine,
            'w_net': w_net,
            'q_in': q_in,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'x_turbine_out': s4.x,
            's_gen_pump': s_gen_pump,
            's_gen_turbine': s_gen_turbine,
            's_gen_boiler': s_gen_boiler,
            's_gen_condenser': s_gen_condenser,
            'exergy_destruction': exergy_destruction,
            'warnings': self.warnings.copy(),
        }
        
        return self.results


class ReheatRankineCycle(BaseCycle):
    """再热Rankine循环
    状态:
      1: 凝出口
      2: 泵出口
      3: 锅炉出口 (高压缸入口)
      4: 高压缸出口 (再热压力P_reheat)
      5: 再热后 (T_reheat)
      6: 低压缸出口
    """
    
    def __init__(self, P_boiler=15, T_boiler=550+273.15, P_cond=0.008,
                 P_reheat=3, T_reheat=550+273.15,
                 eta_pump=0.85, eta_turbine=0.90):
        super().__init__('再热Rankine循环')
        self.P_boiler = P_boiler
        self.T_boiler = T_boiler
        self.P_cond = P_cond
        self.P_reheat = P_reheat
        self.T_reheat = T_reheat
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        self.warnings = []
    
    def compute(self):
        P_b, T_b = self.P_boiler, self.T_boiler
        P_c = self.P_cond
        P_rh, T_rh = self.P_reheat, self.T_reheat
        eta_p, eta_t = self.eta_pump, self.eta_turbine
        
        if T_b > T_MAX_STEAM:
            self.warnings.append(f"锅炉温度过高")
        if T_rh > T_MAX_STEAM:
            self.warnings.append(f"再热温度过高")
        
        # 状态1: 冷凝器出口
        s1 = self.add_state('1', fluid='water', P=P_c, x=0)
        
        # 状态2: 泵出口
        h2_is = s1.h + s1.v * (P_b - P_c) * 1000
        h2_val = s1.h + (h2_is - s1.h) / eta_p
        s2_state = self.add_state('2', fluid='water', P=P_b, h=h2_val)
        
        # 状态3: 锅炉出口
        s3 = self.add_state('3', fluid='water', P=P_b, T=T_b)
        
        # 状态4: 高压缸出口 (等熵到P_reheat)
        s4_is = steam_state(P=P_rh, s=s3.s)
        h4_val = s3.h - eta_t * (s3.h - s4_is['h'])
        s4_state = self.add_state('4', fluid='water', P=P_rh, h=h4_val)
        
        # 状态5: 再热后
        s5 = self.add_state('5', fluid='water', P=P_rh, T=T_rh)
        
        # 状态6: 低压缸出口
        s6_is = steam_state(P=P_c, s=s5.s)
        h6_val = s5.h - eta_t * (s5.h - s6_is['h'])
        s6_state = self.add_state('6', fluid='water', P=P_c, h=h6_val)
        
        if s6_state.x is not None and s6_state.x < MIN_TURBINE_QUALITY:
            self.warnings.append(
                f"末级湿度过大: x={s6_state.x:.4f} < {MIN_TURBINE_QUALITY}"
            )
        
        # 能量计算
        w_pump = h2_val - s1.h
        w_turbine_hp = s3.h - h4_val
        w_turbine_lp = s5.h - h6_val
        w_turbine = w_turbine_hp + w_turbine_lp
        w_net = w_turbine - w_pump
        
        q_in_1 = s3.h - s2_state.h  # 一次加热
        q_in_2 = s5.h - s4_state.h    # 再热
        q_in = q_in_1 + q_in_2
        q_out = h6_val - s1.h
        
        eta = w_net / q_in if q_in > 0 else 0
        
        T_h = max(T_b, T_rh)
        T_l = s1.T
        eta_carnot = 1 - T_l / T_h
        
        # 熵产
        T0 = 298.15
        exergy_destruction = {
            '泵': T0 * (s2_state.s - s1.s),
            '锅炉(一级)': T0 * (s3.s - s2_state.s),
            '高压缸': T0 * (s4_state.s - s3.s),
            '再热器': T0 * (s5.s - s4_state.s),
            '低压缸': T0 * (s6_state.s - s5.s),
            '冷凝器': T0 * (s1.s - s6_state.s + q_out / T0),
        }
        
        self.states['1'] = s1
        self.states['2'] = s2_state
        self.states['3'] = s3
        self.states['4'] = s4_state
        self.states['5'] = s5
        self.states['6'] = s6_state
        
        self.processes = [
            ('1', '2', '泵加压'),
            ('2', '3', '锅炉一级加热'),
            ('3', '4', '高压缸膨胀'),
            ('4', '5', '再热器加热'),
            ('5', '6', '低压缸膨胀'),
            ('6', '1', '冷凝器放热'),
        ]
        
        self.results = {
            'w_pump': w_pump,
            'w_turbine_hp': w_turbine_hp,
            'w_turbine_lp': w_turbine_lp,
            'w_turbine': w_turbine,
            'w_net': w_net,
            'q_in_1': q_in_1,
            'q_in_2': q_in_2,
            'q_in': q_in,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'x_turbine_out': s6_state.x,
            'exergy_destruction': exergy_destruction,
            'warnings': self.warnings.copy(),
        }
        return self.results


class RegenerativeRankineCycle(BaseCycle):
    """回热Rankine循环 - 带一级开式给水加热器
    迭代求解抽汽比例
    """
    
    def __init__(self, P_boiler=12, T_boiler=540+273.15, P_cond=0.008,
                 P_extract=2, eta_pump=0.85, eta_turbine=0.90):
        super().__init__('回热Rankine循环(开式一级)')
        self.P_boiler = P_boiler
        self.T_boiler = T_boiler
        self.P_cond = P_cond
        self.P_extract = P_extract
        self.eta_pump = eta_pump
        self.eta_turbine = eta_turbine
        self.warnings = []
    
    def compute(self):
        P_b, T_b = self.P_boiler, self.T_boiler
        P_c = self.P_cond
        P_ex = self.P_extract
        eta_p, eta_t = self.eta_pump, self.eta_turbine
        
        # 状态1: 冷凝器出口
        s1 = self.add_state('1', fluid='water', P=P_c, x=0)
        
        # 状态2: 一级泵出口 (到加热器压力)
        h2_is = s1.h + s1.v * (P_ex - P_c) * 1000
        h2_val = s1.h + (h2_is - s1.h) / eta_p
        s2_state = self.add_state('2', fluid='water', P=P_ex, h=h2_val)
        
        # 状态3: 锅炉出口
        s3 = self.add_state('3', fluid='water', P=P_b, T=T_b)
        
        # 状态4: 抽汽点 (P_ex)
        s4_is = steam_state(P=P_ex, s=s3.s)
        h4_val = s3.h - eta_t * (s3.h - s4_is['h'])
        s4_state = self.add_state('4', fluid='water', P=P_ex, h=h4_val)
        
        # 状态6: 加热器出口饱和液 (P_ex下饱和液)
        s6_state = self.add_state('6', fluid='water', P=P_ex, x=0)
        h6_val = s6_state.h
        
        # 状态5: 汽轮机出口 (P_cond)
        s5_is = steam_state(P=P_c, s=s4_state.s)
        h5_val = s4_state.h - eta_t * (s4_state.h - s5_is['h'])
        s5_state = self.add_state('5', fluid='water', P=P_c, h=h5_val)
        
        # 迭代求解抽汽比例 y
        # 能量平衡: y*h4 + (1-y)*h2 = h6
        y = (h6_val - s2_state.h) / (h4_val - s2_state.h)
        y = max(0.0, min(0.5, y))
        
        # 收敛判据检查 (直接解,无迭代)
        converged = True
        resid = abs(y * h4_val + (1 - y) * s2_state.h - h6_val) / max(abs(h6_val), 1e-6)
        if resid > 0.001:
            converged = False
        
        # 状态7: 二级泵出口 (锅炉压力)
        h7_is = h6_val + s6_state.v * (P_b - P_ex) * 1000
        h7_val = h6_val + (h7_is - h6_val) / eta_p
        s7_state = self.add_state('7', fluid='water', P=P_b, h=h7_val)
        
        # 干度校验
        if s5_state.x is not None and s5_state.x < MIN_TURBINE_QUALITY:
            self.warnings.append(
                f"末级湿度过大: x={s5_state.x:.4f} < {MIN_TURBINE_QUALITY}"
            )
        
        # 能量计算 (按1kg主蒸汽计)
        w_pump1 = s2_state.h - s1.h
        w_pump2 = h7_val - h6_val
        w_pump = w_pump2 + (1 - y) * w_pump1
        
        w_turbine_hp = s3.h - h4_val
        w_turbine_lp = (1 - y) * (h4_val - h5_val)
        w_turbine = w_turbine_hp + w_turbine_lp
        w_net = w_turbine - w_pump
        
        q_in = s3.h - h7_val
        q_out = (1 - y) * (h5_val - s1.h)
        
        eta = w_net / q_in if q_in > 0 else 0
        
        T_h = T_b
        T_l = s1.T
        eta_carnot = 1 - T_l / T_h
        
        T0 = 298.15
        exergy_destruction = {
            '凝泵': T0 * (1-y) * (s2_state.s - s1.s),
            '给水泵': T0 * (s7_state.s - s6_state.s),
            '加热器': T0 * (s6_state.s - y*s4_state.s - (1-y)*s2_state.s),
            '锅炉': T0 * (s3.s - s7_state.s),
            '高压缸': T0 * (s4_state.s - s3.s),
            '低压缸': T0 * (1-y) * (s5_state.s - s4_state.s),
            '冷凝器': T0 * ((1-y)*(s1.s - s5_state.s) + q_out / T0),
        }
        
        self.states['1'] = s1
        self.states['2'] = s2_state
        self.states['3'] = s3
        self.states['4'] = s4_state
        self.states['5'] = s5_state
        self.states['6'] = s6_state
        self.states['7'] = s7_state
        self.processes = [
            ('1', '2', '凝泵加压'),
            ('2', '6', '加热器混合'),
            ('6', '7', '给水泵加压'),
            ('7', '3', '锅炉加热'),
            ('3', '4', '高压缸膨胀'),
            ('4', '5', '低压缸膨胀'),
            ('5', '1', '冷凝器放热'),
        ]
        
        self.results = {
            'extract_fraction': y,
            'w_pump1': w_pump1,
            'w_pump2': w_pump2,
            'w_pump': w_pump,
            'w_turbine_hp': w_turbine_hp,
            'w_turbine_lp': w_turbine_lp,
            'w_turbine': w_turbine,
            'w_net': w_net,
            'q_in': q_in,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'x_turbine_out': s5_state.x,
            'exergy_destruction': exergy_destruction,
            'converged': converged,
            'residual': resid,
            'warnings': self.warnings.copy(),
        }
        return self.results
