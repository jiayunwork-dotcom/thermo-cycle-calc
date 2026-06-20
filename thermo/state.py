"""
状态点求解器 - 统一的接口
支持水蒸气和理想气体工质
"""

from .steam import steam_state, T_MAX_STEAM
from .ideal_gas import IdealGas, GAS_PARAMS

WORKING_FLUIDS = ['water', 'air', 'nitrogen', 'argon', 'helium', 'co2', 'methane']

class StatePoint:
    """统一的状态点类"""
    
    def __init__(self, fluid='water', label='', **kwargs):
        """
        fluid: 'water' 或 气体名
        **kwargs: 状态参数,如 T=..., P=..., x=..., h=..., s=..., v=...
        """
        self.fluid = fluid
        self.label = label
        self.params = kwargs
        
        if fluid == 'water':
            self._compute_water(**kwargs)
        else:
            self._compute_gas(fluid, **kwargs)
    
    def _compute_water(self, **kwargs):
        result = steam_state(**kwargs)
        self.T = result.get('T')
        self.P = result.get('P')
        self.h = result.get('h')
        self.s = result.get('s')
        self.v = result.get('v')
        self.x = result.get('x')
        self.region = result.get('region')
        self.u = result.get('u', self.h - self.P * 1000 * self.v if self.v else None)
        self.rho = result.get('rho', 1/self.v if self.v else None)
        self._raw = result
    
    def _compute_gas(self, fluid, **kwargs):
        gas = IdealGas(fluid)
        result = gas.state(**kwargs)
        self.gas_obj = gas
        self.T = result.get('T')
        self.P = result.get('P')
        self.h = result.get('h')
        self.s = result.get('s')
        self.v = result.get('v')
        self.u = result.get('u')
        self.x = None
        self.region = 'gas'
        self.rho = result.get('rho', 1/self.v if self.v else None)
        self._raw = result
    
    def to_dict(self):
        return {
            'label': self.label,
            'fluid': self.fluid,
            'T': self.T,
            'P': self.P,
            'h': self.h,
            's': self.s,
            'v': self.v,
            'u': self.u,
            'x': self.x,
            'region': self.region,
            'rho': self.rho,
        }
    
    def summary(self):
        s = f"状态点 {self.label} [{self.fluid}]:\n"
        if self.T is not None:
            s += f"  T = {self.T-273.15:.2f} °C ({self.T:.2f} K)\n"
        if self.P is not None:
            s += f"  P = {self.P:.4f} MPa\n"
        if self.h is not None:
            s += f"  h = {self.h:.2f} kJ/kg\n"
        if self.s is not None:
            s += f"  s = {self.s:.4f} kJ/(kg·K)\n"
        if self.v is not None:
            s += f"  v = {self.v:.6f} m³/kg\n"
        if self.x is not None:
            s += f"  x = {self.x:.4f}\n"
        s += f"  region = {self.region}"
        return s
    
    def __repr__(self):
        return self.summary()


def get_saturation_curve():
    """获取饱和线数据"""
    from .steam import saturation_curve
    return saturation_curve()
