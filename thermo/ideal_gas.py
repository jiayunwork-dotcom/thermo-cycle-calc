"""
理想气体工质属性计算
支持定比热和变比热(温度多项式拟合)
用于燃气循环(Brayton, Otto, Diesel等)
"""

import numpy as np

# 常用气体参数
GAS_PARAMS = {
    'air': {'k': 1.4, 'R': 0.287, 'cp': 1.005, 'cv': 0.718, 
            'T_max': 1500 + 273.15, 'name': '空气'},
    'argon': {'k': 1.667, 'R': 0.2081, 'cp': 0.5203, 'cv': 0.3122,
              'T_max': 2000, 'name': '氩气'},
    'helium': {'k': 1.667, 'R': 2.077, 'cp': 5.193, 'cv': 3.116,
               'T_max': 2000, 'name': '氦气'},
    'nitrogen': {'k': 1.4, 'R': 0.2968, 'cp': 1.039, 'cv': 0.743,
                 'T_max': 1500, 'name': '氮气'},
    'co2': {'k': 1.289, 'R': 0.1889, 'cp': 0.846, 'cv': 0.657,
            'T_max': 1500, 'name': '二氧化碳'},
    'methane': {'k': 1.32, 'R': 0.5182, 'cp': 2.226, 'cv': 1.708,
                'T_max': 1000, 'name': '甲烷'},
}

# 空气的变比热多项式系数 (cp in kJ/(kg·K), T in K)
# 适用范围: 300K ~ 1500K
def cp_air_T(T):
    """空气定压比热随温度变化 kJ/(kg·K)"""
    # 分段多项式拟合
    if T < 600:
        return 1.003 + 5e-5 * (T - 300)
    elif T < 1000:
        return 1.05 + 1.5e-4 * (T - 600)
    else:
        return 1.14 + 8e-5 * (T - 1000)

def cv_air_T(T):
    return cp_air_T(T) - 0.287

def k_air_T(T):
    return cp_air_T(T) / cv_air_T(T)

class IdealGas:
    """理想气体工质类"""
    
    def __init__(self, gas_type='air', k=None, R=None, cp=None, cv=None, T_max=None):
        """
        gas_type: 'air', 'argon', 'nitrogen', 'co2' 等
        或自定义 k, R, cp, cv
        """
        if gas_type in GAS_PARAMS:
            params = GAS_PARAMS[gas_type]
            self.k = params['k']
            self.R = params['R']
            self.cp = params['cp']
            self.cv = params['cv']
            self.T_max = params['T_max']
            self.name = params['name']
            self.gas_type = gas_type
        else:
            self.k = k if k is not None else 1.4
            self.R = R if R is not None else 0.287
            if cp is not None and cv is not None:
                self.cp = cp
                self.cv = cv
            elif cp is not None:
                self.cp = cp
                self.cv = cp - self.R
            elif cv is not None:
                self.cv = cv
                self.cp = cv + self.R
            else:
                self.cp = self.k * self.R / (self.k - 1)
                self.cv = self.R / (self.k - 1)
            self.T_max = T_max if T_max is not None else 1500 + 273.15
            self.name = gas_type
            self.gas_type = 'custom'
    
    def state(self, **kwargs):
        """
        计算理想气体状态
        参数组合:
          T, P - 温度(K), 压力(kPa或MPa,统一用MPa)
          T, v - 温度(K), 比体积(m³/kg)
          P, v - 压力(MPa), 比体积(m³/kg)
          T, h - 温度(K), 比焓(kJ/kg)
          T, s - 温度(K), 比熵(kJ/(kg·K))
          P, h - 压力(MPa), 比焓
          P, s - 压力(MPa), 比熵
        """
        keys = set(kwargs.keys())
        R = self.R
        cp = self.cp
        cv = self.cv
        k = self.k
        
        # 温度校验
        if 'T' in kwargs:
            if kwargs['T'] > self.T_max + 100:
                import warnings
                warnings.warn(f"气体温度超过上限: {kwargs['T']-273.15:.1f}°C > {self.T_max-273.15:.0f}°C")
        
        # T, P
        if 'T' in keys and 'P' in keys:
            T, P = kwargs['T'], kwargs['P']  # P in MPa
            P_kPa = P * 1000
            v = R * T / P_kPa
            h = cp * T
            s = cp * np.log(T / 298.15) - R * np.log(P / 0.1)  # 参考态 25°C, 0.1MPa
            u = cv * T
            return {'T': T, 'P': P, 'v': v, 'h': h, 's': s, 'u': u,
                    'rho': 1/v, 'region': 'gas'}
        
        # T, v
        if 'T' in keys and 'v' in keys:
            T, v = kwargs['T'], kwargs['v']
            P_kPa = R * T / v
            P = P_kPa / 1000
            h = cp * T
            s = cp * np.log(T / 298.15) - R * np.log(P / 0.1)
            u = cv * T
            return {'T': T, 'P': P, 'v': v, 'h': h, 's': s, 'u': u,
                    'rho': 1/v, 'region': 'gas'}
        
        # P, v
        if 'P' in keys and 'v' in keys:
            P, v = kwargs['P'], kwargs['v']
            P_kPa = P * 1000
            T = P_kPa * v / R
            h = cp * T
            s = cp * np.log(T / 298.15) - R * np.log(P / 0.1)
            u = cv * T
            return {'T': T, 'P': P, 'v': v, 'h': h, 's': s, 'u': u,
                    'rho': 1/v, 'region': 'gas'}
        
        # T, h
        if 'T' in keys and 'h' in keys:
            T = kwargs['T']
            h = kwargs['h']
            # 用h反推cp再求P? 这里我们假设定压比热
            # 需要一个额外参数,默认P=0.1MPa
            P = kwargs.get('P', 0.1)
            return self.state(T=T, P=P)
        
        # P, h (等焓过程求T)
        if 'P' in keys and 'h' in keys:
            h = kwargs['h']
            P = kwargs['P']
            T = h / cp
            return self.state(T=T, P=P)
        
        # P, s (等熵过程求T)
        if 'P' in keys and 's' in keys:
            P = kwargs['P']
            s = kwargs['s']
            # s = cp*ln(T/T0) - R*ln(P/P0)
            T0, P0 = 298.15, 0.1
            ln_T_T0 = (s + R * np.log(P / P0)) / cp
            T = T0 * np.exp(ln_T_T0)
            return self.state(T=T, P=P)
        
        # T, s (等熵过程求P)
        if 'T' in keys and 's' in keys:
            T = kwargs['T']
            s = kwargs['s']
            T0, P0 = 298.15, 0.1
            ln_P_P0 = (cp * np.log(T / T0) - s) / R
            P = P0 * np.exp(ln_P_P0)
            return self.state(T=T, P=P)
        
        # h, s
        if 'h' in keys and 's' in keys:
            h, s = kwargs['h'], kwargs['s']
            T = h / cp
            return self.state(T=T, s=s)
        
        raise ValueError(f"不支持的参数组合: {keys}")
    
    # 过程计算
    def isentropic_process(self, state1, P2=None, T2=None, v2=None):
        """
        等熵过程 (k为常数)
        输入: state1 (初始状态dict), 以及终态的一个参数
        """
        k = self.k
        R = self.R
        
        T1, P1, v1 = state1['T'], state1['P'], state1['v']
        
        if P2 is not None:
            T2 = T1 * (P2 / P1) ** ((k - 1) / k)
            v2 = R * T2 / (P2 * 1000)
        elif T2 is not None:
            P2 = P1 * (T2 / T1) ** (k / (k - 1))
            v2 = R * T2 / (P2 * 1000)
        elif v2 is not None:
            T2 = T1 * (v1 / v2) ** (k - 1)
            P2 = R * T2 / (v2 * 1000)
        else:
            raise ValueError("需要P2、T2或v2")
        
        return self.state(T=T2, P=P2)
    
    def polytropic_process(self, state1, n, P2=None, v2=None, T2=None):
        """多变过程 (pv^n=常数)"""
        k = self.k
        R = self.R
        T1, P1, v1 = state1['T'], state1['P'], state1['v']
        
        if P2 is not None:
            T2 = T1 * (P2 / P1) ** ((n - 1) / n)
            v2 = R * T2 / (P2 * 1000)
        elif v2 is not None:
            T2 = T1 * (v1 / v2) ** (n - 1)
            P2 = R * T2 / (v2 * 1000)
        elif T2 is not None:
            v2 = v1 * (T1 / T2) ** (1 / (n - 1))
            P2 = R * T2 / (v2 * 1000)
        else:
            raise ValueError("需要P2、T2或v2")
        
        state2 = self.state(T=T2, P=P2)
        
        # 过程功和热量
        w = (R * (T1 - T2)) / (n - 1) if n != 1 else R * T1 * np.log(v2 / v1)
        q = cv * (T2 - T1) * (k - n) / (k - 1) if n != k else 0
        
        return state2, w, q
    
    def process_with_efficiency(self, state1, P2=None, T2=None, v2=None, 
                                 eta_turbine=None, eta_compressor=None, 
                                 eta_pump=None):
        """
        考虑效率的过程
        涡轮/膨胀机: eta = (h1 - h2_actual) / (h1 - h2_isentropic)
        压缩机/泵:   eta = (h2_isentropic - h1) / (h2_actual - h1)
        """
        state2_is = self.isentropic_process(state1, P2=P2, T2=T2, v2=v2)
        
        h1 = state1['h']
        h2_is = state2_is['h']
        
        if eta_turbine is not None:
            h2_actual = h1 - eta_turbine * (h1 - h2_is)
            return self.state(h=h2_actual, P=state2_is['P']), state2_is
        elif eta_compressor is not None or eta_pump is not None:
            eta = eta_compressor if eta_compressor else eta_pump
            h2_actual = h1 + (h2_is - h1) / eta
            return self.state(h=h2_actual, P=state2_is['P']), state2_is
        else:
            return state2_is, state2_is
