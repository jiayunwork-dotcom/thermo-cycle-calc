"""
参数化分析模块
- 单参数扫描: 效率/净功 vs 参数
- 双参数扫描: 等值线图
"""

import numpy as np
import pandas as pd


def parametric_sweep(cycle_class, cycle_kwargs,
                     param_name, param_min, param_max, n_points=10,
                     param_unit=''):
    """
    单参数扫描
    参数:
      cycle_class: 循环类 (如 RankineCycle)
      cycle_kwargs: 其他固定参数 dict
      param_name: 变化的参数名 (如 'P_boiler')
      param_min, param_max: 参数范围
      n_points: 点数
    返回:
      param_values, eta_values, wnet_values, results_list
    """
    param_values = np.linspace(param_min, param_max, n_points)
    eta_values = []
    wnet_values = []
    results_list = []
    warnings_list = []
    
    for pv in param_values:
        kwargs = cycle_kwargs.copy()
        kwargs[param_name] = pv
        try:
            cycle = cycle_class(**kwargs)
            res = cycle.compute()
            eta_values.append(res.get('eta', 0) or res.get('eta_total', 0))
            wnet_values.append(res.get('w_net', 0) or res.get('W_dot_total_kW', 0))
            results_list.append(res)
            warnings_list.append(res.get('warnings', []))
        except Exception as e:
            eta_values.append(np.nan)
            wnet_values.append(np.nan)
            results_list.append(None)
            warnings_list.append([str(e)])
    
    return (param_values, np.array(eta_values), np.array(wnet_values),
            results_list, warnings_list)


def multi_param_sweep(cycle_class, cycle_kwargs,
                      param1_name, p1_min, p1_max, n1,
                      param2_name, p2_min, p2_max, n2,
                      p1_unit='', p2_unit=''):
    """
    双参数扫描 - 生成等值线数据
    返回:
      p1_vals, p2_vals, eta_mtx (n2 x n1矩阵), wnet_mtx
    """
    p1_vals = np.linspace(p1_min, p1_max, n1)
    p2_vals = np.linspace(p2_min, p2_max, n2)
    
    eta_mtx = np.zeros((n2, n1))
    wnet_mtx = np.zeros((n2, n1))
    
    for i, p2 in enumerate(p2_vals):
        for j, p1 in enumerate(p1_vals):
            kwargs = cycle_kwargs.copy()
            kwargs[param1_name] = p1
            kwargs[param2_name] = p2
            try:
                cycle = cycle_class(**kwargs)
                res = cycle.compute()
                eta_mtx[i, j] = res.get('eta', 0) or res.get('eta_total', 0)
                wnet_mtx[i, j] = res.get('w_net', 0) or res.get('W_dot_total_kW', 0)
            except:
                eta_mtx[i, j] = np.nan
                wnet_mtx[i, j] = np.nan
    
    return p1_vals, p2_vals, eta_mtx, wnet_mtx


def exergy_analysis_summary(results):
    """汇总㶲分析结果"""
    ex_d = results.get('exergy_destruction', {})
    if not ex_d:
        return pd.DataFrame()
    
    total = sum(ex_d.values())
    data = []
    for comp, val in ex_d.items():
        data.append({
            '组件': comp,
            '㶲损失 (kJ/kg)': val,
            '占比 (%)': val / total * 100 if total > 0 else 0,
        })
    
    df = pd.DataFrame(data).sort_values('㶲损失 (kJ/kg)', ascending=False)
    return df
