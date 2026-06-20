"""
燃气-蒸汽联合循环 (CCGT - Combined Cycle Gas Turbine)
Brayton循环的高温排气作为Rankine循环的热源
迭代求解使余热锅炉(HRSG)能量平衡收敛
"""

import numpy as np
from thermo.state import StatePoint
from thermo.ideal_gas import IdealGas
from thermo.steam import steam_state, tsat_P
from cycles.rankine import BaseCycle, RankineCycle, ReheatRankineCycle
from cycles.brayton import BraytonCycle


class CombinedCycle(BaseCycle):
    """燃气-蒸汽联合循环
    
    顶层循环: Brayton (燃气轮机)
    底层循环: Rankine (蒸汽轮机)
    耦合: 燃气排气 → 余热锅炉(HRSG) → 产生蒸汽
    """
    
    def __init__(self, 
                 # 燃气循环参数
                 P1_gas=0.1, T1_gas=25+273.15, rp=14, TIT=1250+273.15,
                 eta_compressor=0.88, eta_turbine_gas=0.92,
                 gas_type='air',
                 # 蒸汽循环参数
                 P_steam=10, T_steam=520+273.15, P_cond=0.008,
                 reheat=True, P_reheat=2.5, T_reheat=520+273.15,
                 eta_pump=0.85, eta_turbine_steam=0.90,
                 # HRSG参数
                 T_stack_min=100+273.15, T_approach=15,
                 m_dot_gas=1.0):
        """
        TIT: 涡轮进口温度 (Turbine Inlet Temperature) K
        T_stack_min: 最低排气温度(避免酸露点) K
        T_approach: 节点温差 K (蒸汽温度与燃气温度差)
        m_dot_gas: 燃气侧质量流量 kg/s
        """
        super().__init__('燃气-蒸汽联合循环CCGT')
        # 燃气
        self.P1_gas = P1_gas
        self.T1_gas = T1_gas
        self.rp = rp
        self.TIT = TIT
        self.eta_compressor = eta_compressor
        self.eta_turbine_gas = eta_turbine_gas
        self.gas_type = gas_type
        # 蒸汽
        self.P_steam = P_steam
        self.T_steam = T_steam
        self.P_cond = P_cond
        self.reheat = reheat
        self.P_reheat = P_reheat
        self.T_reheat = T_reheat
        self.eta_pump = eta_pump
        self.eta_turbine_steam = eta_turbine_steam
        # HRSG
        self.T_stack_min = T_stack_min
        self.T_approach = T_approach
        self.m_dot_gas = m_dot_gas
        self.warnings = []
    
    def compute(self):
        # ---- 第一步: 计算燃气循环 ----
        brayton = BraytonCycle(
            P1=self.P1_gas, T1=self.T1_gas, rp=self.rp, T3=self.TIT,
            eta_compressor=self.eta_compressor,
            eta_turbine=self.eta_turbine_gas,
            gas_type=self.gas_type,
            regenerator=False, intercooler=False
        )
        res_gas = brayton.compute()
        self.gas_cycle = brayton
        
        # 燃气排气状态
        exhaust_label = '4'  # 涡轮出口
        s_gas_exh = brayton.states[exhaust_label]
        T_gas_exh = s_gas_exh.T
        h_gas_exh = s_gas_exh.h
        
        gas = IdealGas(self.gas_type)
        cp_gas = gas.cp
        
        # ---- 第二步: 迭代求解蒸汽循环蒸发量 ----
        # 目标: 燃气从T_gas_exh降到T_stack,加热水从凝水温度到T_steam
        
        # 饱和温度
        T_sat_steam = tsat_P(self.P_steam)
        
        # 估计排气最终温度(不低于T_stack_min,也不低于T_sat_steam + T_approach)
        T_stack_target = max(self.T_stack_min, T_sat_steam + self.T_approach)
        
        # 蒸汽循环
        if self.reheat:
            steam_cycle = ReheatRankineCycle(
                P_boiler=self.P_steam, T_boiler=self.T_steam,
                P_cond=self.P_cond,
                P_reheat=self.P_reheat, T_reheat=self.T_reheat,
                eta_pump=self.eta_pump, eta_turbine=self.eta_turbine_steam
            )
        else:
            steam_cycle = RankineCycle(
                P_boiler=self.P_steam, T_boiler=self.T_steam,
                P_cond=self.P_cond,
                eta_pump=self.eta_pump, eta_turbine=self.eta_turbine_steam
            )
        res_steam = steam_cycle.compute()
        self.steam_cycle = steam_cycle
        
        # 蒸汽侧需要的热量 (每kg蒸汽)
        if self.reheat:
            q_per_kg_steam = res_steam['q_in']  # 一级+再热
        else:
            q_per_kg_steam = res_steam['q_in']
        
        # 燃气侧可提供的热量 (每kg燃气)
        q_available_per_kg_gas = cp_gas * (T_gas_exh - T_stack_target)
        
        # 质量流量比 m_dot_steam / m_dot_gas
        if q_per_kg_steam > 0:
            mass_ratio = q_available_per_kg_gas / q_per_kg_steam
        else:
            mass_ratio = 0.1
        
        # 迭代修正: 能量平衡
        # HRSG: 燃气降温 = 蒸汽吸热
        # 实际排气温度
        T_stack_actual = T_gas_exh - mass_ratio * q_per_kg_steam / cp_gas
        T_stack_actual = max(T_stack_actual, self.T_stack_min)
        
        # 迭代收敛判据
        for i in range(20):
            q_from_gas = cp_gas * (T_gas_exh - T_stack_actual)
            mass_ratio_new = q_from_gas / q_per_kg_steam if q_per_kg_steam > 0 else mass_ratio
            
            # 相对误差
            err = abs(mass_ratio_new - mass_ratio) / max(mass_ratio, 1e-6)
            mass_ratio = mass_ratio_new
            
            T_stack_actual = T_gas_exh - mass_ratio * q_per_kg_steam / cp_gas
            T_stack_actual = max(T_stack_actual, self.T_stack_min)
            
            if err < 0.001:
                break
        
        # ---- 第三步: 计算整体性能 ----
        m_dot_steam = self.m_dot_gas * mass_ratio
        
        # 净功
        w_net_gas = res_gas['w_net']  # kJ/kg gas
        w_net_steam = res_steam['w_net']  # kJ/kg steam
        
        W_dot_gas = self.m_dot_gas * w_net_gas  # kW
        W_dot_steam = m_dot_steam * w_net_steam  # kW
        W_dot_total = W_dot_gas + W_dot_steam
        
        # 总输入热量 (燃气燃烧室)
        q_in_gas = res_gas['q_in']  # kJ/kg gas
        Q_dot_in = self.m_dot_gas * q_in_gas  # kW
        
        # 效率
        eta_gas = res_gas['eta']
        eta_steam = res_steam['eta']
        eta_total = W_dot_total / Q_dot_in if Q_dot_in > 0 else 0
        
        # 理论联合效率公式: 1 - (1-η_gas)(1-η_steam)
        eta_combined_theory = 1 - (1 - eta_gas) * (1 - eta_steam)
        
        # Carnot效率
        T_h = self.TIT
        T_l = steam_cycle.states.get('1', steam_cycle.states.get('6')).T
        eta_carnot = 1 - T_l / T_h
        
        # 合并状态点
        for label, sp in brayton.states.items():
            new_label = f'G{label}'
            # 用实际T和P构造
            try:
                sp_copy = StatePoint(fluid=sp.fluid, label=new_label, T=sp.T, P=sp.P)
            except:
                sp_copy = StatePoint(fluid=sp.fluid, label=new_label, P=sp.P, h=sp.h)
            # 拷贝其他属性
            if sp.h: sp_copy.h = sp.h
            if sp.s: sp_copy.s = sp.s
            if sp.v: sp_copy.v = sp.v
            sp_copy.x, sp_copy.region = sp.x, sp.region
            sp_copy._raw = sp._raw
            self.states[new_label] = sp_copy
        
        for label, sp in steam_cycle.states.items():
            new_label = f'S{label}'
            try:
                if sp.x is not None:
                    sp_copy = StatePoint(fluid=sp.fluid, label=new_label, P=sp.P, x=sp.x)
                else:
                    sp_copy = StatePoint(fluid=sp.fluid, label=new_label, T=sp.T, P=sp.P)
            except:
                sp_copy = StatePoint(fluid=sp.fluid, label=new_label, P=sp.P, h=sp.h)
            if sp.h: sp_copy.h = sp.h
            if sp.s: sp_copy.s = sp.s
            if sp.v: sp_copy.v = sp.v
            sp_copy.x, sp_copy.region = sp.x, sp.region
            sp_copy._raw = sp._raw
            self.states[new_label] = sp_copy
        
        # 㶲损失合并
        exergy_destruction = {}
        T0 = 298.15
        for k, v in res_gas.get('exergy_destruction', {}).items():
            exergy_destruction[f'燃气-{k}'] = v
        for k, v in res_steam.get('exergy_destruction', {}).items():
            exergy_destruction[f'蒸汽-{k}'] = v
        # HRSG㶲损失
        s_gas_out = gas.state(T=T_stack_actual, P=s_gas_exh.P).get('s', s_gas_exh.s)
        s_gen_hrsg = self.m_dot_gas * (s_gas_out - s_gas_exh.s) + \
                     m_dot_steam * (steam_cycle.states.get('3', steam_cycle.states.get('S3')).s - 
                                  steam_cycle.states.get('2', steam_cycle.states.get('S2')).s)
        # 简化HRSG熵产
        exergy_destruction['余热锅炉(HRSG)'] = T0 * abs(q_from_gas) * \
            (1/T_stack_actual - 1/T_gas_exh) if T_gas_exh > T_stack_actual else 0
        
        self.warnings = res_gas.get('warnings', []) + res_steam.get('warnings', [])
        
        self.results = {
            'eta_gas': eta_gas,
            'eta_steam': eta_steam,
            'eta_total': eta_total,
            'eta_combined_theory': eta_combined_theory,
            'eta_carnot': eta_carnot,
            'W_dot_gas_kW': W_dot_gas,
            'W_dot_steam_kW': W_dot_steam,
            'W_dot_total_kW': W_dot_total,
            'Q_dot_in_kW': Q_dot_in,
            'mass_ratio_steam_gas': mass_ratio,
            'm_dot_steam_kg_s': m_dot_steam,
            'm_dot_gas_kg_s': self.m_dot_gas,
            'T_gas_exhaust_C': T_gas_exh - 273.15,
            'T_stack_C': T_stack_actual - 273.15,
            'T_stack_min_C': self.T_stack_min - 273.15,
            'w_net_gas': w_net_gas,
            'w_net_steam': w_net_steam,
            'converged_iterations': i + 1,
            'exergy_destruction': exergy_destruction,
            'gas_results': res_gas,
            'steam_results': res_steam,
            'warnings': self.warnings,
        }
        return self.results
