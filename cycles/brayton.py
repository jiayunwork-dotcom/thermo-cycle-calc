"""
Brayton循环 (燃气轮机循环)
- 基础Brayton循环
- 带回热器的Brayton循环
- 带中间冷却的多级压缩Brayton循环
"""

import numpy as np
from thermo.state import StatePoint
from thermo.ideal_gas import IdealGas
from cycles.rankine import BaseCycle


class BraytonCycle(BaseCycle):
    """基础Brayton循环
    状态:
      1: 压气机入口 (T1, P1)
      2: 压气机出口
      3: 燃烧室出口 (T3, P3≈P2)
      4: 涡轮出口 (P4≈P1)
    """
    
    def __init__(self, P1=0.1, T1=25+273.15, rp=10, T3=1100+273.15,
                 eta_compressor=0.85, eta_turbine=0.90,
                 gas_type='air', regenerator=False, eta_regenerator=0.85,
                 intercooler=False, P_inter=None, T_inter_out=None):
        """
        P1: 入口压力 MPa
        T1: 入口温度 K
        rp: 压比
        T3: 涡轮前温度 K
        eta_compressor: 压气机效率
        eta_turbine: 涡轮效率
        regenerator: 是否带回热器
        eta_regenerator: 回热器效率
        intercooler: 是否中间冷却
        P_inter: 中间压力 MPa (None则取几何平均)
        T_inter_out: 中冷后温度 K (None则=T1)
        """
        super().__init__('燃气轮机Brayton循环')
        self.P1 = P1
        self.T1 = T1
        self.rp = rp
        self.T3 = T3
        self.eta_compressor = eta_compressor
        self.eta_turbine = eta_turbine
        self.gas_type = gas_type
        self.regenerator = regenerator
        self.eta_regenerator = eta_regenerator
        self.intercooler = intercooler
        self.P_inter = P_inter
        self.T_inter_out = T_inter_out if T_inter_out else T1
        self.warnings = []
    
    def compute(self):
        P1, T1 = self.P1, self.T1
        rp = self.rp
        T3 = self.T3
        eta_c = self.eta_compressor
        eta_t = self.eta_turbine
        
        gas = IdealGas(self.gas_type)
        k, cp = gas.k, gas.cp
        
        # 温度校验
        if T3 > gas.T_max:
            self.warnings.append(
                f"涡轮前温度{T3-273.15:.1f}°C超过上限{gas.T_max-273.15:.0f}°C"
            )
        
        if self.intercooler:
            return self._compute_intercooled(gas)
        
        if self.regenerator:
            return self._compute_regenerative(gas)
        
        # ---- 基础循环 ----
        P2 = P1 * rp
        P4 = P1
        
        # 状态1
        s1 = self.add_state('1', fluid=self.gas_type, T=T1, P=P1)
        
        # 状态2s: 等熵压缩
        s2s, _ = gas.process_with_efficiency(s1._raw, P2=P2, eta_compressor=1.0)
        h2s = s2s['h']
        h2 = s1.h + (h2s - s1.h) / eta_c
        s2 = self.add_state('2', fluid=self.gas_type, P=P2, h=h2)
        
        # 状态3
        s3 = self.add_state('3', fluid=self.gas_type, T=T3, P=P2)
        
        # 状态4s: 等熵膨胀
        s4s, _ = gas.process_with_efficiency(s3._raw, P2=P4, eta_turbine=1.0)
        h4s = s4s['h']
        h4 = s3.h - eta_t * (s3.h - h4s)
        s4 = self.add_state('4', fluid=self.gas_type, P=P4, h=h4)
        
        # 能量计算
        w_comp = s2.h - s1.h
        w_turb = s3.h - s4.h
        w_net = w_turb - w_comp
        q_in = s3.h - s2.h
        q_out = s4.h - s1.h
        
        eta = w_net / q_in if q_in > 0 else 0
        
        # Carnot效率
        T_h = T3
        T_l = T1
        eta_carnot = 1 - T_l / T_h
        
        T0 = 298.15
        exergy_destruction = {
            '压气机': T0 * (s2.s - s1.s),
            '燃烧室': T0 * (s3.s - s2.s),
            '涡轮': T0 * (s4.s - s3.s),
            '排气放热': T0 * (s1.s - s4.s + q_out / T0),
        }
        
        self.processes = [
            ('1', '2', '压气机压缩'),
            ('2', '3', '燃烧室等压加热'),
            ('3', '4', '涡轮膨胀做功'),
            ('4', '1', '排气等压放热'),
        ]
        
        self.results = {
            'w_compressor': w_comp,
            'w_turbine': w_turb,
            'w_net': w_net,
            'q_in': q_in,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'back_work_ratio': w_comp / w_turb if w_turb > 0 else 0,
            'exergy_destruction': exergy_destruction,
            'warnings': self.warnings.copy(),
        }
        return self.results
    
    def _compute_regenerative(self, gas):
        """带回热器的Brayton循环"""
        P1, T1 = self.P1, self.T1
        rp = self.rp
        T3 = self.T3
        eta_c = self.eta_compressor
        eta_t = self.eta_turbine
        eta_r = self.eta_regenerator
        
        k, cp = gas.k, gas.cp
        P2 = P1 * rp
        P4 = P1
        
        # 状态1
        s1 = self.add_state('1', fluid=self.gas_type, T=T1, P=P1)
        
        # 状态2: 压气机出口
        s2s, _ = gas.process_with_efficiency(s1._raw, P2=P2, eta_compressor=1.0)
        h2 = s1.h + (s2s['h'] - s1.h) / eta_c
        s2 = self.add_state('2', fluid=self.gas_type, P=P2, h=h2)
        
        # 状态5: 涡轮出口 (未回热)
        s3 = self.add_state('3', fluid=self.gas_type, T=T3, P=P2)
        s5s, _ = gas.process_with_efficiency(s3._raw, P2=P4, eta_turbine=1.0)
        h5 = s3.h - eta_t * (s3.h - s5s['h'])
        s5 = self.add_state('5', fluid=self.gas_type, P=P4, h=h5)
        
        # 回热器: 用排气(5)预热压缩空气(2→4)
        # 回热器效率: eta_r = (T4 - T2) / (T5 - T2)
        T2 = s2.T
        T5_val = s5.T
        T4 = T2 + eta_r * (T5_val - T2)
        s4 = self.add_state('4', fluid=self.gas_type, P=P2, T=T4)
        
        # 排气冷却到环境 (5→6→1, 6为回热后排气)
        T6 = T5_val - (T4 - T2)  # 能量守恒
        s6 = self.add_state('6', fluid=self.gas_type, P=P4, T=T6)
        
        # 状态3不变 (燃烧室出口)
        # 能量计算
        w_comp = s2.h - s1.h
        w_turb = s3.h - s5.h
        w_net = w_turb - w_comp
        
        # 外部加热: 从4→3
        q_in = s3.h - s4.h
        # 外部放热: 从6→1
        q_out = s6.h - s1.h
        
        eta = w_net / q_in if q_in > 0 else 0
        
        T_h = T3
        T_l = T1
        eta_carnot = 1 - T_l / T_h
        
        T0 = 298.15
        exergy_destruction = {
            '压气机': T0 * (s2.s - s1.s),
            '回热器': T0 * (s4.s + s6.s - s2.s - s5.s),
            '燃烧室': T0 * (s3.s - s4.s),
            '涡轮': T0 * (s5.s - s3.s),
            '排气放热': T0 * (s1.s - s6.s + q_out / T0),
        }
        
        self.processes = [
            ('1', '2', '压气机压缩'),
            ('2', '4', '回热器预热'),
            ('4', '3', '燃烧室加热'),
            ('3', '5', '涡轮膨胀'),
            ('5', '6', '回热器放热'),
            ('6', '1', '排气放热'),
        ]
        
        self.results = {
            'w_compressor': w_comp,
            'w_turbine': w_turb,
            'w_net': w_net,
            'q_in': q_in,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'back_work_ratio': w_comp / w_turb if w_turb > 0 else 0,
            'T_regen_out': T4,
            'T_exhaust_after_regen': T6,
            'exergy_destruction': exergy_destruction,
            'warnings': self.warnings.copy(),
        }
        return self.results
    
    def _compute_intercooled(self, gas):
        """带中间冷却的多级压缩"""
        P1, T1 = self.P1, self.T1
        rp = self.rp
        T3 = self.T3
        eta_c = self.eta_compressor
        eta_t = self.eta_turbine
        k, cp = gas.k, gas.cp
        
        # 中间压力 (几何平均)
        if self.P_inter:
            P_inter = self.P_inter
        else:
            P_inter = P1 * np.sqrt(rp)
        
        P2 = P1 * rp
        P4 = P1
        
        # 状态1
        s1 = self.add_state('1', fluid=self.gas_type, T=T1, P=P1)
        
        # 状态2: 低压压气机出口
        s2s, _ = gas.process_with_efficiency(s1._raw, P2=P_inter, eta_compressor=1.0)
        h2 = s1.h + (s2s['h'] - s1.h) / eta_c
        s2 = self.add_state('2', fluid=self.gas_type, P=P_inter, h=h2)
        
        # 状态3: 中间冷却后
        T_ic_out = self.T_inter_out
        s3 = self.add_state('3', fluid=self.gas_type, T=T_ic_out, P=P_inter)
        
        # 状态4: 高压压气机出口
        s4s, _ = gas.process_with_efficiency(s3._raw, P2=P2, eta_compressor=1.0)
        h4 = s3.h + (s4s['h'] - s3.h) / eta_c
        s4 = self.add_state('4', fluid=self.gas_type, P=P2, h=h4)
        
        # 状态5: 燃烧室出口
        s5 = self.add_state('5', fluid=self.gas_type, T=T3, P=P2)
        
        # 状态6: 涡轮出口
        s6s, _ = gas.process_with_efficiency(s5._raw, P2=P4, eta_turbine=1.0)
        h6 = s5.h - eta_t * (s5.h - s6s['h'])
        s6 = self.add_state('6', fluid=self.gas_type, P=P4, h=h6)
        
        # 能量计算
        w_comp_lp = s2.h - s1.h
        w_comp_hp = s4.h - s3.h
        w_comp = w_comp_lp + w_comp_hp
        w_turb = s5.h - s6.h
        w_net = w_turb - w_comp
        
        q_in = s5.h - s4.h
        q_out_ic = s2.h - s3.h  # 中冷放热量
        q_out_exh = s6.h - s1.h  # 排气放热
        q_out = q_out_ic + q_out_exh
        
        eta = w_net / q_in if q_in > 0 else 0
        
        T_h = T3
        T_l = T1
        eta_carnot = 1 - T_l / T_h
        
        T0 = 298.15
        exergy_destruction = {
            '低压压气机': T0 * (s2.s - s1.s),
            '中间冷却器': T0 * (s3.s - s2.s + q_out_ic / T0),
            '高压压气机': T0 * (s4.s - s3.s),
            '燃烧室': T0 * (s5.s - s4.s),
            '涡轮': T0 * (s6.s - s5.s),
            '排气放热': T0 * (s1.s - s6.s + q_out_exh / T0),
        }
        
        self.processes = [
            ('1', '2', '低压压气机'),
            ('2', '3', '中间冷却'),
            ('3', '4', '高压压气机'),
            ('4', '5', '燃烧室加热'),
            ('5', '6', '涡轮膨胀'),
            ('6', '1', '排气放热'),
        ]
        
        self.results = {
            'w_comp_lp': w_comp_lp,
            'w_comp_hp': w_comp_hp,
            'w_compressor': w_comp,
            'w_turbine': w_turb,
            'w_net': w_net,
            'q_in': q_in,
            'q_out_ic': q_out_ic,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'back_work_ratio': w_comp / w_turb if w_turb > 0 else 0,
            'exergy_destruction': exergy_destruction,
            'warnings': self.warnings.copy(),
        }
        return self.results
