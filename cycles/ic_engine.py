"""
活塞式内燃机循环
- Otto循环 (等容加热)
- Diesel循环 (等压加热)
"""

import numpy as np
from thermo.state import StatePoint
from thermo.ideal_gas import IdealGas
from cycles.rankine import BaseCycle


class OttoCycle(BaseCycle):
    """Otto循环 (汽油机, 等容加热循环)
    状态:
      1: 进气终点 (压缩起点)
      2: 压缩终点 (燃烧前)
      3: 燃烧终点 (等容加热后)
      4: 膨胀终点 (排气前)
    """
    
    def __init__(self, T1=25+273.15, P1=0.1, r=8, q_in=1800,
                 eta_compression=0.90, eta_expansion=0.90, gas_type='air'):
        """
        T1: 压缩起点温度 K
        P1: 压缩起点压力 MPa
        r: 压缩比 V1/V2
        q_in: 单位质量加热量 kJ/kg
        eta_compression: 压缩过程效率
        eta_expansion: 膨胀过程效率
        """
        super().__init__('Otto循环(汽油机)')
        self.T1 = T1
        self.P1 = P1
        self.r = r
        self.q_in = q_in
        self.eta_compression = eta_compression
        self.eta_expansion = eta_expansion
        self.gas_type = gas_type
        self.warnings = []
    
    def compute(self):
        T1, P1 = self.T1, self.P1
        r = self.r
        q_in = self.q_in
        
        gas = IdealGas(self.gas_type)
        k, cp, cv = gas.k, gas.cp, gas.cv
        
        # 状态1
        s1 = self.add_state('1', fluid=self.gas_type, T=T1, P=P1)
        v1 = s1.v
        
        # 状态2: 压缩终点 (v2 = v1/r)
        v2 = v1 / r
        # 等熵压缩终点
        T2s = T1 * r ** (k - 1)
        P2s = P1 * r ** k
        s2s = gas.state(T=T2s, P=P2s)
        w_comp_is = s2s['u'] - s1.u
        
        # 实际压缩
        w_comp_actual = w_comp_is / self.eta_compression
        u2 = s1.u + w_comp_actual
        T2 = u2 / cv
        P2 = gas.R * T2 / (v2 * 1000)
        s2 = self.add_state('2', fluid=self.gas_type, T=T2, P=P2)
        
        # 温度校验
        T3 = T2 + q_in / cv
        if T3 > gas.T_max:
            self.warnings.append(
                f"燃烧后温度{T3-273.15:.1f}°C超过上限{gas.T_max-273.15:.0f}°C"
            )
        
        # 状态3: 燃烧终点 (等容加热)
        P3 = P2 * (T3 / T2)
        s3 = self.add_state('3', fluid=self.gas_type, T=T3, P=P3)
        
        # 状态4: 膨胀终点 (v4 = v1)
        v4 = v1
        # 等熵膨胀
        T4s = T3 * (1 / r) ** (k - 1)
        P4s = P3 * (1 / r) ** k
        s4s = gas.state(T=T4s, P=P4s)
        w_exp_is = s3.u - s4s['u']
        
        # 实际膨胀
        w_exp_actual = w_exp_is * self.eta_expansion
        u4 = s3.u - w_exp_actual
        T4 = u4 / cv
        P4 = gas.R * T4 / (v4 * 1000)
        s4 = self.add_state('4', fluid=self.gas_type, T=T4, P=P4)
        
        # 放热量 (等容放热 4→1)
        q_out = u4 - s1.u
        
        w_net = q_in - q_out
        eta = w_net / q_in if q_in > 0 else 0
        
        # Carnot效率
        T_h = T3
        T_l = T1
        eta_carnot = 1 - T_l / T_h
        
        # 平均有效压力
        v_displacement = v1 - v2
        mep = w_net / v_displacement / 1000  # MPa (w_net kJ/kg, v m³/kg)
        
        T0 = 298.15
        exergy_destruction = {
            '压缩过程': T0 * (s2.s - s1.s),
            '燃烧过程': T0 * (s3.s - s2.s),
            '膨胀过程': T0 * (s4.s - s3.s),
            '排气放热': T0 * (s1.s - s4.s + q_out / T0),
        }
        
        self.processes = [
            ('1', '2', '压缩过程'),
            ('2', '3', '等容燃烧'),
            ('3', '4', '膨胀做功'),
            ('4', '1', '排气放热'),
        ]
        
        self.results = {
            'w_compression': w_comp_actual,
            'w_expansion': w_exp_actual,
            'w_net': w_net,
            'q_in': q_in,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'eta_otto_ideal': 1 - 1 / r ** (k - 1),
            'mep': mep,
            'T_max': T3,
            'P_max': P3,
            'exergy_destruction': exergy_destruction,
            'warnings': self.warnings.copy(),
        }
        return self.results


class DieselCycle(BaseCycle):
    """Diesel循环 (柴油机, 等压加热循环)
    状态:
      1: 压缩起点
      2: 压缩终点
      3: 燃烧终点 (等压加热结束)
      4: 膨胀终点
    """
    
    def __init__(self, T1=25+273.15, P1=0.1, r=16, cutoff=2.0,
                 eta_compression=0.92, eta_expansion=0.92, gas_type='air'):
        """
        cutoff: 定压预胀比 (v3/v2)
        r: 压缩比
        """
        super().__init__('Diesel循环(柴油机)')
        self.T1 = T1
        self.P1 = P1
        self.r = r
        self.cutoff = cutoff
        self.eta_compression = eta_compression
        self.eta_expansion = eta_expansion
        self.gas_type = gas_type
        self.warnings = []
    
    def compute(self):
        T1, P1 = self.T1, self.P1
        r = self.r
        cutoff = self.cutoff
        
        gas = IdealGas(self.gas_type)
        k, cp, cv, R = gas.k, gas.cp, gas.cv, gas.R
        
        # 状态1
        s1 = self.add_state('1', fluid=self.gas_type, T=T1, P=P1)
        v1 = s1.v
        
        # 状态2: 压缩终点
        v2 = v1 / r
        T2s = T1 * r ** (k - 1)
        P2s = P1 * r ** k
        s2s = gas.state(T=T2s, P=P2s)
        w_comp_is = s2s['u'] - s1.u
        
        w_comp_actual = w_comp_is / self.eta_compression
        u2 = s1.u + w_comp_actual
        T2 = u2 / cv
        P2 = R * T2 / (v2 * 1000)
        s2 = self.add_state('2', fluid=self.gas_type, T=T2, P=P2)
        
        # 状态3: 等压加热终点 (v3 = cutoff * v2)
        P3 = P2
        v3 = cutoff * v2
        T3 = P3 * 1000 * v3 / R
        s3 = self.add_state('3', fluid=self.gas_type, T=T3, P=P3)
        
        q_in = cp * (T3 - T2)
        
        if T3 > gas.T_max:
            self.warnings.append(
                f"燃烧后温度{T3-273.15:.1f}°C超过上限{gas.T_max-273.15:.0f}°C"
            )
        
        # 状态4: 膨胀终点 (v4 = v1)
        v4 = v1
        # 膨胀比 v4/v3
        re = v4 / v3
        T4s = T3 * (1 / re) ** (k - 1)
        P4s = P3 * (1 / re) ** k
        s4s = gas.state(T=T4s, P=P4s)
        w_exp_is = s3.u - s4s['u']
        
        w_exp_actual = w_exp_is * self.eta_expansion
        u4 = s3.u - w_exp_actual
        T4 = u4 / cv
        P4 = R * T4 / (v4 * 1000)
        s4 = self.add_state('4', fluid=self.gas_type, T=T4, P=P4)
        
        # 放热量 (等容放热)
        q_out = cv * (T4 - T1)
        
        w_net = q_in - q_out
        eta = w_net / q_in if q_in > 0 else 0
        
        # 理想Diesel效率
        eta_diesel_ideal = 1 - (cutoff**k - 1) / (k * r**(k-1) * (cutoff - 1))
        
        T_h = T3
        T_l = T1
        eta_carnot = 1 - T_l / T_h
        
        v_displacement = v1 - v2
        mep = w_net / v_displacement / 1000
        
        T0 = 298.15
        exergy_destruction = {
            '压缩过程': T0 * (s2.s - s1.s),
            '燃烧过程': T0 * (s3.s - s2.s),
            '膨胀过程': T0 * (s4.s - s3.s),
            '排气放热': T0 * (s1.s - s4.s + q_out / T0),
        }
        
        self.processes = [
            ('1', '2', '压缩过程'),
            ('2', '3', '等压燃烧'),
            ('3', '4', '膨胀做功'),
            ('4', '1', '排气放热'),
        ]
        
        self.results = {
            'w_compression': w_comp_actual,
            'w_expansion': w_exp_actual,
            'w_net': w_net,
            'q_in': q_in,
            'q_out': q_out,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'eta_diesel_ideal': eta_diesel_ideal,
            'mep': mep,
            'T_max': T3,
            'P_max': P3,
            'cutoff': cutoff,
            'exergy_destruction': exergy_destruction,
            'warnings': self.warnings.copy(),
        }
        return self.results
