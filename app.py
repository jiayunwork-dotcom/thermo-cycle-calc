"""
工程热力学循环分析工具 - Dash交互面板
作者: ThermoCycleCalc
功能: Rankine/Brayton/Otto/Diesel/联合循环 计算与可视化
"""

import os
import sys
import json
import base64
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

# 确保路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cycles.rankine import RankineCycle, ReheatRankineCycle, RegenerativeRankineCycle
from cycles.brayton import BraytonCycle
from cycles.ic_engine import OttoCycle, DieselCycle
from cycles.combined import CombinedCycle

from plots import (
    plot_Ts_diagram, plot_Pv_diagram, plot_hs_diagram,
    plot_exergy_bar, plot_parametric_curve, plot_2d_contour
)
from analysis import parametric_sweep, multi_param_sweep, exergy_analysis_summary
from export_utils import (
    export_states_csv, export_results_csv,
    export_figure_svg, export_figure_png,
    generate_pdf_report
)
from thermo.state import StatePoint, WORKING_FLUIDS
from thermo.steam import steam_state, psat_T, tsat_P

app = dash.Dash(__name__, 
                title='工程热力学循环分析工具',
                suppress_callback_exceptions=True)
server = app.server

# ============================================================
# 循环参数配置定义
# ============================================================
CYCLE_CONFIGS = {
    'rankine_basic': {
        'name': '基础Rankine循环',
        'class': RankineCycle,
        'params': [
            {'key': 'P_boiler', 'label': '锅炉压力', 'unit': 'MPa', 'min': 1, 'max': 22, 'step': 0.5, 'default': 10},
            {'key': 'T_boiler', 'label': '锅炉出口温度', 'unit': '°C', 'min': 200, 'max': 650, 'step': 10, 'default': 500},
            {'key': 'P_cond', 'label': '冷凝器压力', 'unit': 'MPa', 'min': 0.003, 'max': 0.2, 'step': 0.001, 'default': 0.01},
            {'key': 'eta_pump', 'label': '泵效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.85},
            {'key': 'eta_turbine', 'label': '汽轮机效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.90},
        ]
    },
    'rankine_reheat': {
        'name': '再热Rankine循环',
        'class': ReheatRankineCycle,
        'params': [
            {'key': 'P_boiler', 'label': '锅炉压力', 'unit': 'MPa', 'min': 3, 'max': 22, 'step': 0.5, 'default': 15},
            {'key': 'T_boiler', 'label': '锅炉出口温度', 'unit': '°C', 'min': 300, 'max': 650, 'step': 10, 'default': 550},
            {'key': 'P_reheat', 'label': '再热压力', 'unit': 'MPa', 'min': 0.5, 'max': 8, 'step': 0.1, 'default': 3},
            {'key': 'T_reheat', 'label': '再热温度', 'unit': '°C', 'min': 300, 'max': 650, 'step': 10, 'default': 550},
            {'key': 'P_cond', 'label': '冷凝器压力', 'unit': 'MPa', 'min': 0.003, 'max': 0.2, 'step': 0.001, 'default': 0.008},
            {'key': 'eta_pump', 'label': '泵效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.85},
            {'key': 'eta_turbine', 'label': '汽轮机效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.90},
        ]
    },
    'rankine_regen': {
        'name': '回热Rankine循环(开式一级)',
        'class': RegenerativeRankineCycle,
        'params': [
            {'key': 'P_boiler', 'label': '锅炉压力', 'unit': 'MPa', 'min': 3, 'max': 22, 'step': 0.5, 'default': 12},
            {'key': 'T_boiler', 'label': '锅炉出口温度', 'unit': '°C', 'min': 300, 'max': 650, 'step': 10, 'default': 540},
            {'key': 'P_extract', 'label': '抽汽压力', 'unit': 'MPa', 'min': 0.3, 'max': 6, 'step': 0.1, 'default': 2},
            {'key': 'P_cond', 'label': '冷凝器压力', 'unit': 'MPa', 'min': 0.003, 'max': 0.2, 'step': 0.001, 'default': 0.008},
            {'key': 'eta_pump', 'label': '泵效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.85},
            {'key': 'eta_turbine', 'label': '汽轮机效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.90},
        ]
    },
    'brayton_basic': {
        'name': '基础Brayton循环(燃气轮机)',
        'class': BraytonCycle,
        'params': [
            {'key': 'rp', 'label': '压比', 'unit': '', 'min': 2, 'max': 40, 'step': 1, 'default': 10},
            {'key': 'T3', 'label': '涡轮进口温度(TIT)', 'unit': '°C', 'min': 600, 'max': 1500, 'step': 20, 'default': 1100},
            {'key': 'P1', 'label': '入口压力', 'unit': 'MPa', 'min': 0.05, 'max': 0.5, 'step': 0.01, 'default': 0.1},
            {'key': 'T1', 'label': '入口温度', 'unit': '°C', 'min': -20, 'max': 60, 'step': 1, 'default': 25},
            {'key': 'eta_compressor', 'label': '压气机效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.85},
            {'key': 'eta_turbine', 'label': '涡轮效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.90},
        ]
    },
    'brayton_regen': {
        'name': '回热Brayton循环',
        'class': BraytonCycle,
        'extra_kwargs': {'regenerator': True},
        'params': [
            {'key': 'rp', 'label': '压比', 'unit': '', 'min': 2, 'max': 30, 'step': 1, 'default': 8},
            {'key': 'T3', 'label': '涡轮进口温度', 'unit': '°C', 'min': 600, 'max': 1500, 'step': 20, 'default': 1000},
            {'key': 'P1', 'label': '入口压力', 'unit': 'MPa', 'min': 0.05, 'max': 0.5, 'step': 0.01, 'default': 0.1},
            {'key': 'T1', 'label': '入口温度', 'unit': '°C', 'min': -20, 'max': 60, 'step': 1, 'default': 25},
            {'key': 'eta_regenerator', 'label': '回热器效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.85},
            {'key': 'eta_compressor', 'label': '压气机效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.85},
            {'key': 'eta_turbine', 'label': '涡轮效率', 'unit': '', 'min': 0.5, 'max': 1.0, 'step': 0.01, 'default': 0.90},
        ]
    },
    'otto': {
        'name': 'Otto循环(汽油机)',
        'class': OttoCycle,
        'params': [
            {'key': 'r', 'label': '压缩比', 'unit': '', 'min': 4, 'max': 14, 'step': 0.5, 'default': 8},
            {'key': 'q_in', 'label': '加热量', 'unit': 'kJ/kg', 'min': 500, 'max': 3000, 'step': 50, 'default': 1800},
            {'key': 'T1', 'label': '压缩起点温度', 'unit': '°C', 'min': 0, 'max': 80, 'step': 1, 'default': 25},
            {'key': 'P1', 'label': '压缩起点压力', 'unit': 'MPa', 'min': 0.05, 'max': 0.2, 'step': 0.005, 'default': 0.1},
            {'key': 'eta_compression', 'label': '压缩效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.90},
            {'key': 'eta_expansion', 'label': '膨胀效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.90},
        ]
    },
    'diesel': {
        'name': 'Diesel循环(柴油机)',
        'class': DieselCycle,
        'params': [
            {'key': 'r', 'label': '压缩比', 'unit': '', 'min': 10, 'max': 25, 'step': 0.5, 'default': 16},
            {'key': 'cutoff', 'label': '预胀比', 'unit': '', 'min': 1.2, 'max': 4, 'step': 0.1, 'default': 2.0},
            {'key': 'T1', 'label': '压缩起点温度', 'unit': '°C', 'min': 0, 'max': 80, 'step': 1, 'default': 25},
            {'key': 'P1', 'label': '压缩起点压力', 'unit': 'MPa', 'min': 0.05, 'max': 0.2, 'step': 0.005, 'default': 0.1},
            {'key': 'eta_compression', 'label': '压缩效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.92},
            {'key': 'eta_expansion', 'label': '膨胀效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.92},
        ]
    },
    'ccgt': {
        'name': '燃气-蒸汽联合循环(CCGT)',
        'class': CombinedCycle,
        'params': [
            {'key': 'rp', 'label': '燃机压比', 'unit': '', 'min': 5, 'max': 30, 'step': 1, 'default': 14},
            {'key': 'TIT', 'label': '燃机涡轮进口温度', 'unit': '°C', 'min': 800, 'max': 1500, 'step': 20, 'default': 1250},
            {'key': 'P_steam', 'label': '蒸汽锅炉压力', 'unit': 'MPa', 'min': 3, 'max': 20, 'step': 0.5, 'default': 10},
            {'key': 'T_steam', 'label': '蒸汽温度', 'unit': '°C', 'min': 350, 'max': 600, 'step': 10, 'default': 520},
            {'key': 'P_cond', 'label': '冷凝器压力', 'unit': 'MPa', 'min': 0.003, 'max': 0.1, 'step': 0.001, 'default': 0.008},
            {'key': 'eta_compressor', 'label': '压气机效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.88},
            {'key': 'eta_turbine_gas', 'label': '燃气涡轮效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.92},
            {'key': 'eta_turbine_steam', 'label': '蒸汽轮机效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.90},
            {'key': 'eta_pump', 'label': '泵效率', 'unit': '', 'min': 0.6, 'max': 1.0, 'step': 0.01, 'default': 0.85},
        ]
    },
}


def build_cycle(cfg_key, param_values):
    """根据配置和参数值构建循环对象并计算"""
    cfg = CYCLE_CONFIGS[cfg_key]
    cls = cfg['class']
    
    kwargs = {}
    for p in cfg['params']:
        val = param_values.get(p['key'], p['default'])
        # 温度转换: °C → K
        if p['key'].startswith('T'):
            kwargs[p['key']] = val + 273.15
        else:
            kwargs[p['key']] = val
    
    # 额外固定参数
    if 'extra_kwargs' in cfg:
        kwargs.update(cfg['extra_kwargs'])
    
    cycle = cls(**kwargs)
    results = cycle.compute()
    return cycle, results


def build_param_inputs(cfg_key):
    """动态构建参数输入组件"""
    cfg = CYCLE_CONFIGS.get(cfg_key)
    if not cfg:
        return []
    
    children = []
    for p in cfg['params']:
        children.append(html.Div([
            html.Label(f"{p['label']} ({p['unit']})" if p['unit'] else p['label'],
                      style={'fontSize': 13, 'marginBottom': 4, 'display': 'block'}),
            dcc.Input(
                id=f"param-{cfg_key}-{p['key']}",
                type='number',
                min=p['min'], max=p['max'], step=p['step'],
                value=p['default'],
                style={'width': '100%', 'padding': 6, 'borderRadius': 4,
                       'border': '1px solid #ccc', 'fontSize': 13}
            ),
        ], style={'marginBottom': 14}))
    return children


# ============================================================
# 页面布局
# ============================================================
app.layout = html.Div([
    # 标题栏
    html.Div([
        html.H1('🔥 工程热力学循环分析与热效率计算工具', 
                style={'margin': 0, 'color': 'white', 'fontSize': 24}),
        html.Div('基于IAPWS-IF97水蒸气模型 & 理想气体模型 | Dash + Plotly',
                style={'color': '#ecf0f1', 'fontSize': 13, 'marginTop': 4}),
    ], style={
        'background': 'linear-gradient(135deg, #2c3e50, #34495e)',
        'padding': '16px 24px', 'borderBottom': '3px solid #e74c3c'
    }),
    
    # 主体: 左侧参数面板 + 右侧结果
    html.Div([
        # ===== 左侧参数面板 =====
        html.Div([
            # 循环选择
            html.Div([
                html.Label('选择循环类型:', style={'fontWeight': 'bold', 'fontSize': 14,
                                                     'display': 'block', 'marginBottom': 8}),
                dcc.Dropdown(
                    id='cycle-type',
                    options=[{'label': v['name'], 'value': k} 
                             for k, v in CYCLE_CONFIGS.items()],
                    value='rankine_basic',
                    clearable=False,
                    style={'fontSize': 14}
                ),
            ], style={'marginBottom': 18, 'padding': 12, 'background': '#f8f9fa',
                     'borderRadius': 8, 'border': '1px solid #dee2e6'}),
            
            # 参数输入区(预生成所有循环类型面板,用display控制显示)
            html.Div([
                html.Div(
                    build_param_inputs(ck),
                    id=f'param-panel-{ck}',
                    style={'display': ('block' if ck == 'rankine_basic' else 'none')}
                ) for ck in CYCLE_CONFIGS
            ], style={'padding': 12, 'background': 'white',
                      'borderRadius': 8, 'border': '1px solid #dee2e6',
                      'marginBottom': 18, 'maxHeight': '500px', 'overflowY': 'auto'}),
            
            # 图表选择
            html.Div([
                html.Label('热力学图类型:', style={'fontWeight': 'bold', 
                                                     'display': 'block', 'marginBottom': 8}),
                dcc.RadioItems(
                    id='diagram-type',
                    options=[
                        {'label': ' T-s 图 (温度-比熵)', 'value': 'Ts'},
                        {'label': ' P-v 图 (压力-比体积)', 'value': 'Pv'},
                        {'label': ' h-s 图 (Mollier图)', 'value': 'hs'},
                        {'label': ' 㶲损失分布', 'value': 'exergy'},
                    ],
                    value='Ts',
                    labelStyle={'display': 'block', 'marginBottom': 6, 'fontSize': 13}
                ),
            ], style={'padding': 12, 'background': 'white',
                     'borderRadius': 8, 'border': '1px solid #dee2e6',
                     'marginBottom': 18}),
            
            # 计算按钮 + 导出
            html.Div([
                html.Button('🔄 重新计算', id='btn-compute', n_clicks=0,
                           style={'background': '#e74c3c', 'color': 'white',
                                  'border': 'none', 'padding': '10px 20px',
                                  'borderRadius': 6, 'fontSize': 14, 'fontWeight': 'bold',
                                  'cursor': 'pointer', 'width': '100%', 'marginBottom': 8}),
                html.Div([
                    html.Button('📄 导出PDF报告', id='btn-export-pdf', n_clicks=0,
                               style={'background': '#27ae60', 'color': 'white',
                                      'border': 'none', 'padding': '8px 12px',
                                      'borderRadius': 4, 'fontSize': 12, 'cursor': 'pointer',
                                      'width': '48%', 'marginRight': '4%'}),
                    html.Button('📊 导出SVG图', id='btn-export-svg', n_clicks=0,
                               style={'background': '#2980b9', 'color': 'white',
                                      'border': 'none', 'padding': '8px 12px',
                                      'borderRadius': 4, 'fontSize': 12, 'cursor': 'pointer',
                                      'width': '48%'}),
                ], style={'display': 'flex', 'marginBottom': 8}),
                html.Div([
                    html.Button('📋 导出CSV(状态点)', id='btn-export-csv-states', n_clicks=0,
                               style={'background': '#8e44ad', 'color': 'white',
                                      'border': 'none', 'padding': '8px 12px',
                                      'borderRadius': 4, 'fontSize': 12, 'cursor': 'pointer',
                                      'width': '100%'}),
                ]),
                dcc.Download(id='download-file'),
            ], style={'marginBottom': 18}),
            
            # 状态点查询工具
            html.Div([
                html.Label('🧪 状态点快速查询 (水蒸气)', 
                          style={'fontWeight': 'bold', 'display': 'block', 
                                 'marginBottom': 8, 'fontSize': 13}),
                html.Div([
                    html.Div([
                        html.Label('T (°C):', style={'fontSize': 12}),
                        dcc.Input(id='sp-T', type='number', value=500,
                                  style={'width': '60px', 'padding': 3, 'fontSize': 12}),
                    ], style={'display': 'inline-block', 'marginRight': 10}),
                    html.Div([
                        html.Label('P (MPa):', style={'fontSize': 12}),
                        dcc.Input(id='sp-P', type='number', value=10, step=0.1,
                                  style={'width': '60px', 'padding': 3, 'fontSize': 12}),
                    ], style={'display': 'inline-block'}),
                ]),
                html.Div(id='sp-result', style={'marginTop': 8, 'fontSize': 12,
                                                 'background': '#f8f9fa',
                                                 'padding': 8, 'borderRadius': 4}),
            ], style={'padding': 12, 'background': 'white',
                     'borderRadius': 8, 'border': '1px solid #dee2e6'}),
            
        ], style={'width': '320px', 'padding': 16, 'background': '#ecf0f1',
                  'minHeight': 'calc(100vh - 80px)'}),
        
        # ===== 右侧结果区 =====
        html.Div([
            # 警告信息
            html.Div(id='warnings-box', style={'marginBottom': 12}),
            
            # 效率卡片
            html.Div(id='efficiency-cards',
                    style={'display': 'flex', 'gap': 12, 'marginBottom': 16}),
            
            # Tabs
            dcc.Tabs(id='tabs', value='tab-diagram', children=[
                dcc.Tab(label='📈 热力学图', value='tab-diagram',
                       style={'padding': 10, 'fontWeight': 'bold'}),
                dcc.Tab(label='📋 状态点参数', value='tab-states',
                       style={'padding': 10, 'fontWeight': 'bold'}),
                dcc.Tab(label='🔍 第二定律分析', value='tab-exergy',
                       style={'padding': 10, 'fontWeight': 'bold'}),
                dcc.Tab(label='📊 参数化分析', value='tab-parametric',
                       style={'padding': 10, 'fontWeight': 'bold'}),
            ], style={'marginBottom': 12}),
            
            html.Div(id='tab-content'),
            
        ], style={'flex': 1, 'padding': 16, 'background': 'white'}),
        
    ], style={'display': 'flex'}),
    
    # 隐藏的存储
    dcc.Store(id='cycle-data-store'),
    
], style={'fontFamily': 'PingFang SC, Microsoft YaHei, Arial, sans-serif'})


# ============================================================
# 回调函数
# ============================================================

# 动态参数面板切换(通过display控制显示隐藏)
@app.callback(
    [Output(f'param-panel-{ck}', 'style') for ck in CYCLE_CONFIGS],
    Input('cycle-type', 'value')
)
def update_param_panel_visibility(cycle_type):
    result = []
    for ck in CYCLE_CONFIGS:
        if ck == cycle_type:
            result.append({'display': 'block'})
        else:
            result.append({'display': 'none'})
    return result


# 主计算: 收集所有参数并计算
def _collect_params(cycle_type, ctx=None):
    """从参数面板收集所有值"""
    cfg = CYCLE_CONFIGS[cycle_type]
    values = {}
    for p in cfg['params']:
        input_id = f"param-{cycle_type}-{p['key']}"
        # 通过State收集
        pass
    return values


# 效率卡片 & 图表 & 状态点 & 警告
@app.callback(
    [Output('efficiency-cards', 'children'),
     Output('warnings-box', 'children'),
     Output('cycle-data-store', 'data')],
    [Input('cycle-type', 'value'),
     Input('btn-compute', 'n_clicks')] +
    [Input(f"param-{k}-{p['key']}", 'value') 
     for k in CYCLE_CONFIGS 
     for p in CYCLE_CONFIGS[k]['params']],
    prevent_initial_call=False
)
def compute_cycle(cycle_type, n_clicks, *all_params):
    """主计算逻辑"""
    cfg = CYCLE_CONFIGS[cycle_type]
    
    # 提取对应循环类型的参数
    param_dict = {}
    idx = 0
    for ck in CYCLE_CONFIGS:
        for p in CYCLE_CONFIGS[ck]['params']:
            if ck == cycle_type and idx < len(all_params):
                if all_params[idx] is not None:
                    param_dict[p['key']] = all_params[idx]
                else:
                    param_dict[p['key']] = p['default']
            idx += 1
    
    try:
        cycle, res = build_cycle(cycle_type, param_dict)
    except Exception as e:
        return (
            [html.Div(f"❌ 计算错误: {e}", 
                     style={'background': '#fdecea', 'color': '#c0392b',
                            'padding': 20, 'borderRadius': 8, 'fontWeight': 'bold'})],
            [], None
        )
    
    # 效率卡片
    eta = res.get('eta', res.get('eta_total', 0))
    eta_carnot = res.get('eta_carnot', 0)
    w_net = res.get('w_net', res.get('W_dot_total_kW', 0))
    q_in = res.get('q_in', 0)
    w_net_unit = 'kJ/kg' if 'w_net' in res else 'kW'
    
    def make_card(title, value, color, unit=''):
        return html.Div([
            html.Div(title, style={'fontSize': 12, 'color': '#7f8c8d', 'marginBottom': 4}),
            html.Div(f'{value}{unit}', 
                    style={'fontSize': 22, 'fontWeight': 'bold', 'color': color}),
        ], style={'flex': 1, 'background': '#f8f9fa', 'padding': 14, 'borderRadius': 8,
                  'borderLeft': f'4px solid {color}'})
    
    cards = [
        make_card('热效率 η', f'{eta*100:.2f}', '#e74c3c', '%'),
        make_card('Carnot效率', f'{eta_carnot*100:.2f}', '#2980b9', '%'),
        make_card('净输出功', f'{w_net:.2f}', '#27ae60', f' {w_net_unit}'),
        make_card('吸热量', f'{q_in:.2f}', '#f39c12', ' kJ/kg' if 'w_net' in res else ' kW'),
    ]
    
    # 联合循环额外卡片
    if 'eta_gas' in res:
        cards.append(make_card('燃气效率', f"{res['eta_gas']*100:.2f}", '#8e44ad', '%'))
        cards.append(make_card('蒸汽效率', f"{res['eta_steam']*100:.2f}", '#16a085', '%'))
    
    # 警告
    warnings = res.get('warnings', [])
    warning_box = []
    if warnings:
        warning_box = [html.Div([
            html.Div('⚠️ 警告:', style={'fontWeight': 'bold', 'color': '#e67e22'}),
            html.Ul([html.Li(w, style={'color': '#d35400'}) for w in warnings])
        ], style={'background': '#fef5e7', 'border': '1px solid #f5b041',
                  'padding': 12, 'borderRadius': 8})]
    
    # 保存数据用于后续
    store_data = {
        'cycle_type': cycle_type,
        'results': _make_serializable(res),
    }
    
    return cards, warning_box, store_data


def _make_serializable(obj):
    """将结果转为可JSON序列化格式"""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


# Tab内容渲染
@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value'),
               Input('cycle-data-store', 'data'),
               Input('diagram-type', 'value')])
def render_tab(tab, store_data, diagram_type):
    if not store_data:
        return html.Div('加载中...')
    
    cycle_type = store_data['cycle_type']
    cfg = CYCLE_CONFIGS[cycle_type]
    
    # 重新构建循环以获取StatePoint对象
    # 简化方法: 重新计算 (因为Store无法存复杂对象)
    ctx = dash.callback_context
    param_values = {}
    for p in cfg['params']:
        param_values[p['key']] = p['default']
    
    try:
        cycle, res = build_cycle(cycle_type, param_values)
    except:
        return html.Div('无法重新构建循环数据')
    
    if tab == 'tab-diagram':
        return _render_diagram_tab(cycle, res, diagram_type)
    elif tab == 'tab-states':
        return _render_states_tab(cycle, res)
    elif tab == 'tab-exergy':
        return _render_exergy_tab(cycle, res)
    elif tab == 'tab-parametric':
        return _render_parametric_tab(cycle_type)
    else:
        return html.Div()


def _render_diagram_tab(cycle, res, diagram_type):
    """渲染热力学图"""
    if diagram_type == 'Ts':
        fig = plot_Ts_diagram(cycle)
    elif diagram_type == 'Pv':
        fig = plot_Pv_diagram(cycle)
    elif diagram_type == 'hs':
        fig = plot_hs_diagram(cycle)
    elif diagram_type == 'exergy':
        fig = plot_exergy_bar(res)
    else:
        fig = go.Figure()
    
    return html.Div([
        dcc.Graph(id='main-diagram', figure=fig,
                 style={'height': '600px', 'width': '100%'})
    ])


def _render_states_tab(cycle, res):
    """渲染状态点参数表"""
    rows = []
    for label, sp in sorted(cycle.states.items()):
        rows.append({
            '状态点': label,
            '温度 (°C)': round(sp.T - 273.15, 2) if sp.T else None,
            '压力 (MPa)': round(sp.P, 4) if sp.P else None,
            '比焓 (kJ/kg)': round(sp.h, 2) if sp.h else None,
            '比熵 (kJ/kg·K)': round(sp.s, 4) if sp.s else None,
            '比体积 (m³/kg)': round(sp.v, 6) if sp.v else None,
            '干度 x': round(sp.x, 4) if sp.x is not None else '-',
            '区域': str(sp.region),
        })
    
    df = pd.DataFrame(rows)
    
    # 关键结果
    summary_rows = []
    key_items = [
        ('热效率 η', 'eta', '%'),
        ('Carnot效率', 'eta_carnot', '%'),
        ('净功', 'w_net', 'kJ/kg'),
        ('吸热量', 'q_in', 'kJ/kg'),
        ('放热量', 'q_out', 'kJ/kg'),
    ]
    for label, key, unit in key_items:
        if key in res and res[key] is not None:
            val = res[key]
            if unit == '%':
                val = f'{val*100:.3f}%'
            else:
                val = f'{val:.3f} {unit}'
            summary_rows.append({'项目': label, '数值': val})
    
    summary_df = pd.DataFrame(summary_rows)
    
    return html.Div([
        html.H4('📋 循环性能汇总', style={'marginTop': 0}),
        dash_table.DataTable(
            data=summary_df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in summary_df.columns],
            style_table={'marginBottom': 20, 'width': '50%'},
            style_header={'backgroundColor': '#3498db', 'color': 'white',
                         'fontWeight': 'bold'},
            style_cell={'padding': '8px 12px', 'fontSize': 13},
        ),
        html.H4('🔧 状态点参数', style={'marginTop': 0}),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in df.columns],
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': '#27ae60', 'color': 'white',
                         'fontWeight': 'bold'},
            style_cell={'padding': '6px 10px', 'fontSize': 12,
                       'textAlign': 'center'},
            style_data_conditional=[
                {'if': {'column_id': '状态点'},
                 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}
            ]
        ),
    ])


def _render_exergy_tab(cycle, res):
    """渲染㶲分析"""
    fig = plot_exergy_bar(res)
    
    ex_df = exergy_analysis_summary(res)
    
    total_ex = sum(res.get('exergy_destruction', {}).values())
    w_net = res.get('w_net', res.get('W_dot_total_kW', 0))
    q_in = res.get('q_in', 0)
    T0 = 298.15
    T_h = 1500  # 近似
    exergy_input = q_in * (1 - T0 / T_h) if q_in else 0
    exergy_eff = w_net / exergy_input * 100 if exergy_input > 0 else 0
    
    return html.Div([
        html.Div([
            html.Div([
                html.Div('㶲效率 (第二定律效率)', 
                        style={'fontSize': 12, 'color': '#7f8c8d'}),
                html.Div(f'{exergy_eff:.2f}%', 
                        style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#8e44ad'}),
            ], style={'display': 'inline-block', 'marginRight': 40}),
            html.Div([
                html.Div('总㶲损失', 
                        style={'fontSize': 12, 'color': '#7f8c8d'}),
                html.Div(f'{total_ex:.3f} kJ/kg', 
                        style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#e74c3c'}),
            ], style={'display': 'inline-block'}),
        ], style={'marginBottom': 20, 'padding': 16, 'background': '#f8f9fa',
                 'borderRadius': 8}),
        dcc.Graph(figure=fig, style={'height': '450px'}),
        html.H4('各组件㶲损失明细'),
        dash_table.DataTable(
            data=ex_df.to_dict('records') if not ex_df.empty else [],
            columns=[{'name': c, 'id': c} for c in (ex_df.columns if not ex_df.empty else [])],
            style_table={'width': '60%'},
            style_header={'backgroundColor': '#e67e22', 'color': 'white',
                         'fontWeight': 'bold'},
            style_cell={'padding': '6px 10px', 'textAlign': 'center'},
        ) if not ex_df.empty else html.Div('无㶲损失数据'),
    ])


def _render_parametric_tab(cycle_type):
    """渲染参数化分析界面"""
    cfg = CYCLE_CONFIGS[cycle_type]
    param_options = [{'label': f"{p['label']} ({p['unit']})" if p['unit'] else p['label'],
                      'value': p['key']} for p in cfg['params']]
    
    return html.Div([
        html.H4('📊 参数化分析', style={'marginTop': 0}),
        
        dcc.Tabs(id='sweep-mode-tabs', value='sweep-single', children=[
            dcc.Tab(label='单参数扫描', value='sweep-single'),
            dcc.Tab(label='双参数扫描(等值线图)', value='sweep-double'),
        ]),
        html.Div(id='sweep-mode-content', style={'marginTop': 16}),
    ])


@app.callback(Output('sweep-mode-content', 'children'),
              [Input('sweep-mode-tabs', 'value'),
               Input('cycle-type', 'value')])
def _render_sweep_mode_content(mode, cycle_type):
    cfg = CYCLE_CONFIGS[cycle_type]
    param_options = [{'label': f"{p['label']} ({p['unit']})" if p['unit'] else p['label'],
                      'value': p['key']} for p in cfg['params']]
    
    if mode == 'sweep-single':
        return html.Div([
            html.H5('📈 单参数扫描', style={'marginTop': 0}),
            html.Div([
                html.Div([
                    html.Label('选择扫描参数:'),
                    dcc.Dropdown(id='sweep-param', options=param_options,
                                value=cfg['params'][0]['key'], clearable=False),
                ], style={'width': '30%', 'display': 'inline-block', 'marginRight': 20}),
                html.Div([
                    html.Label('最小值:'),
                    dcc.Input(id='sweep-min', type='number', value=cfg['params'][0]['min'],
                             style={'width': '100px'}),
                ], style={'display': 'inline-block', 'marginRight': 20}),
                html.Div([
                    html.Label('最大值:'),
                    dcc.Input(id='sweep-max', type='number', value=cfg['params'][0]['max'],
                             style={'width': '100px'}),
                ], style={'display': 'inline-block', 'marginRight': 20}),
                html.Div([
                    html.Label('点数:'),
                    dcc.Input(id='sweep-n', type='number', value=10, min=3, max=50,
                             style={'width': '80px'}),
                ], style={'display': 'inline-block', 'marginRight': 20}),
                html.Button('开始扫描', id='btn-sweep', n_clicks=0,
                           style={'background': '#2980b9', 'color': 'white',
                                  'border': 'none', 'padding': '8px 16px',
                                  'borderRadius': 4, 'cursor': 'pointer'}),
            ], style={'marginBottom': 20, 'padding': 16, 'background': '#f8f9fa',
                     'borderRadius': 8}),
            
            dcc.Loading(
                id='loading-sweep',
                type='default',
                children=html.Div(id='sweep-results')
            ),
        ])
    else:  # 双参数扫描
        return html.Div([
            html.H5('🗺 双参数扫描 (等值线图)', style={'marginTop': 0}),
            html.Div([
                # 参数X
                html.Div([
                    html.Label('参数 X (横轴):'),
                    dcc.Dropdown(id='sweep2-param-x', options=param_options,
                                value=cfg['params'][0]['key'], clearable=False),
                    html.Div([
                        html.Label('最小值:'),
                        dcc.Input(id='sweep2-x-min', type='number', value=cfg['params'][0]['min'],
                                 style={'width': '80px'}),
                        html.Label('  最大值:'),
                        dcc.Input(id='sweep2-x-max', type='number', value=cfg['params'][0]['max'],
                                 style={'width': '80px'}),
                        html.Label('  点数:'),
                        dcc.Input(id='sweep2-x-n', type='number', value=8, min=3, max=20,
                                 style={'width': '60px'}),
                    ], style={'marginTop': 6}),
                ], style={'width': '47%', 'display': 'inline-block', 'marginRight': '3%',
                         'padding': 12, 'background': '#eaf2f8', 'borderRadius': 6}),
                # 参数Y
                html.Div([
                    html.Label('参数 Y (纵轴):'),
                    dcc.Dropdown(id='sweep2-param-y', options=param_options,
                                value=cfg['params'][1]['key'] if len(cfg['params'])>1 else cfg['params'][0]['key'], 
                                clearable=False),
                    html.Div([
                        html.Label('最小值:'),
                        dcc.Input(id='sweep2-y-min', type='number', 
                                 value=cfg['params'][1]['min'] if len(cfg['params'])>1 else cfg['params'][0]['min'],
                                 style={'width': '80px'}),
                        html.Label('  最大值:'),
                        dcc.Input(id='sweep2-y-max', type='number', 
                                 value=cfg['params'][1]['max'] if len(cfg['params'])>1 else cfg['params'][0]['max'],
                                 style={'width': '80px'}),
                        html.Label('  点数:'),
                        dcc.Input(id='sweep2-y-n', type='number', value=8, min=3, max=20,
                                 style={'width': '60px'}),
                    ], style={'marginTop': 6}),
                ], style={'width': '47%', 'display': 'inline-block',
                         'padding': 12, 'background': '#fdf2e9', 'borderRadius': 6}),
            ], style={'marginBottom': 12}),
            
            html.Div([
                html.Label('输出指标: '),
                dcc.RadioItems(
                    id='sweep2-z',
                    options=[
                        {'label': ' 热效率 η', 'value': 'eta'},
                        {'label': ' 净输出功 w_net', 'value': 'wnet'},
                    ],
                    value='eta',
                    labelStyle={'display': 'inline-block', 'marginRight': 20}
                ),
                html.Button('开始扫描', id='btn-sweep2', n_clicks=0,
                           style={'background': '#8e44ad', 'color': 'white',
                                  'border': 'none', 'padding': '8px 16px',
                                  'borderRadius': 4, 'cursor': 'pointer',
                                  'marginLeft': 20}),
            ], style={'marginBottom': 20, 'padding': 16, 'background': '#f8f9fa',
                     'borderRadius': 8}),
            
            dcc.Loading(
                id='loading-sweep2',
                type='default',
                children=html.Div(id='sweep2-results')
            ),
        ])


# 参数化扫描执行
@app.callback(Output('sweep-results', 'children'),
              [Input('btn-sweep', 'n_clicks'),
               Input('cycle-type', 'value')],
              [State('sweep-param', 'value'),
               State('sweep-min', 'value'),
               State('sweep-max', 'value'),
               State('sweep-n', 'value')])
def run_parametric_sweep(n_clicks, cycle_type, param, pmin, pmax, n):
    if n_clicks == 0 or not param:
        raise PreventUpdate
    
    cfg = CYCLE_CONFIGS[cycle_type]
    cls = cfg['class']
    
    # 默认参数
    kwargs = {}
    for p in cfg['params']:
        if p['key'].startswith('T'):
            kwargs[p['key']] = p['default'] + 273.15
        else:
            kwargs[p['key']] = p['default']
    
    if 'extra_kwargs' in cfg:
        kwargs.update(cfg['extra_kwargs'])
    
    # 温度转换
    pmin_calc = pmin + 273.15 if param.startswith('T') else pmin
    pmax_calc = pmax + 273.15 if param.startswith('T') else pmax
    
    try:
        pvals, etas, wnets, _, _ = parametric_sweep(
            cls, kwargs, param, pmin_calc, pmax_calc, n
        )
    except Exception as e:
        return html.Div(f'扫描失败: {e}', style={'color': '#e74c3c'})
    
    # 转换回来
    if param.startswith('T'):
        pvals = pvals - 273.15
    
    p_cfg = next((p for p in cfg['params'] if p['key'] == param), None)
    p_label = p_cfg['label'] if p_cfg else param
    p_unit = p_cfg['unit'] if p_cfg else ''
    
    fig = plot_parametric_curve(pvals, etas, wnets, p_label, p_unit,
                                title=f'{cfg["name"]} - 参数化分析')
    
    return html.Div([
        dcc.Graph(figure=fig, style={'height': '500px'}),
    ])


# 双参数扫描执行
@app.callback(Output('sweep2-results', 'children'),
              [Input('btn-sweep2', 'n_clicks'),
               Input('cycle-type', 'value')],
              [State('sweep2-param-x', 'value'),
               State('sweep2-x-min', 'value'),
               State('sweep2-x-max', 'value'),
               State('sweep2-x-n', 'value'),
               State('sweep2-param-y', 'value'),
               State('sweep2-y-min', 'value'),
               State('sweep2-y-max', 'value'),
               State('sweep2-y-n', 'value'),
               State('sweep2-z', 'value')])
def run_double_sweep(n_clicks, cycle_type, px, px_min, px_max, px_n,
                     py, py_min, py_max, py_n, z_metric):
    if n_clicks == 0 or not px or not py:
        raise PreventUpdate
    
    cfg = CYCLE_CONFIGS[cycle_type]
    cls = cfg['class']
    
    # 默认参数
    kwargs = {}
    for p in cfg['params']:
        if p['key'].startswith('T'):
            kwargs[p['key']] = p['default'] + 273.15
        else:
            kwargs[p['key']] = p['default']
    
    if 'extra_kwargs' in cfg:
        kwargs.update(cfg['extra_kwargs'])
    
    # 温度转换
    def _conv(param, val):
        return val + 273.15 if param.startswith('T') else val
    
    px_min_c = _conv(px, px_min)
    px_max_c = _conv(px, px_max)
    py_min_c = _conv(py, py_min)
    py_max_c = _conv(py, py_max)
    
    try:
        p1_vals, p2_vals, eta_mtx, wnet_mtx = multi_param_sweep(
            cls, kwargs, px, px_min_c, px_max_c, px_n,
            py, py_min_c, py_max_c, py_n
        )
    except Exception as e:
        return html.Div(f'扫描失败: {e}', style={'color': '#e74c3c'})
    
    # 温度参数转回°C
    if px.startswith('T'):
        p1_vals = p1_vals - 273.15
    if py.startswith('T'):
        p2_vals = p2_vals - 273.15
    
    px_cfg = next((p for p in cfg['params'] if p['key'] == px), None)
    py_cfg = next((p for p in cfg['params'] if p['key'] == py), None)
    px_label = px_cfg['label'] if px_cfg else px
    py_label = py_cfg['label'] if py_cfg else py
    px_unit = px_cfg['unit'] if px_cfg else ''
    py_unit = py_cfg['unit'] if py_cfg else ''
    
    if z_metric == 'eta':
        z_mtx = eta_mtx * 100
        z_name = '热效率'
        z_unit = '%'
    else:
        z_mtx = wnet_mtx
        z_name = '净输出功'
        z_unit = 'kJ/kg'
    
    fig = plot_2d_contour(p1_vals, p2_vals, z_mtx,
                          x_name=px_label, y_name=py_label, z_name=z_name,
                          x_unit=px_unit, y_unit=py_unit, z_unit=z_unit,
                          title=f'{cfg["name"]} - 双参数扫描 ({z_name})')
    
    return html.Div([
        html.Div([
            html.Strong(f'网格: {px_n} × {py_n} = {px_n * py_n} 个计算点'),
            html.Span(f'  |  X: {px_label} [{px_min}, {px_max}] {px_unit}'),
            html.Span(f'  |  Y: {py_label} [{py_min}, {py_max}] {py_unit}'),
        ], style={'marginBottom': 10, 'padding': 10, 'background': '#f0f3f4',
                  'borderRadius': 4, 'fontSize': 12}),
        dcc.Graph(figure=fig, style={'height': '550px'}),
    ])


# 状态点快速查询
@app.callback(Output('sp-result', 'children'),
              [Input('sp-T', 'value'), Input('sp-P', 'value')])
def query_state_point(T_val, P_val):
    if T_val is None or P_val is None:
        return '请输入T和P'
    try:
        T_K = T_val + 273.15
        st = steam_state(T=T_K, P=P_val)
        x_txt = f"x={st.get('x', '-')}" if st.get('x') is not None else ''
        region_name = {1: '压缩液', 2: '过热蒸气', 3: '两相区'}.get(st['region'], '?')
        return (
            f"[{region_name}] "
            f"h={st.get('h', 0):.2f} kJ/kg, "
            f"s={st.get('s', 0):.4f} kJ/(kg·K), "
            f"v={st.get('v', 0):.6f} m³/kg {x_txt}"
        )
    except Exception as e:
        return f'计算失败: {e}'


# 导出功能
@app.callback(Output('download-file', 'data'),
              [Input('btn-export-pdf', 'n_clicks'),
               Input('btn-export-svg', 'n_clicks'),
               Input('btn-export-csv-states', 'n_clicks')],
              [State('cycle-type', 'value'),
               State('diagram-type', 'value')] +
              [State(f"param-{k}-{p['key']}", 'value') 
               for k in CYCLE_CONFIGS 
               for p in CYCLE_CONFIGS[k]['params']],
              prevent_initial_call=True)
def handle_export(n_pdf, n_svg, n_csv, cycle_type, diagram_type, *all_params):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    cfg = CYCLE_CONFIGS[cycle_type]
    
    # 收集用户实际参数
    param_dict = {}
    idx = 0
    for ck in CYCLE_CONFIGS:
        for p in CYCLE_CONFIGS[ck]['params']:
            if ck == cycle_type and idx < len(all_params):
                if all_params[idx] is not None:
                    param_dict[p['key']] = all_params[idx]
                else:
                    param_dict[p['key']] = p['default']
            idx += 1
    
    # 温度参数转K
    kwargs = {}
    for k, v in param_dict.items():
        if k.startswith('T_'):
            kwargs[k] = v + 273.15
        else:
            kwargs[k] = v
    if 'extra_kwargs' in cfg:
        kwargs.update(cfg['extra_kwargs'])
    
    cycle = cfg['class'](**kwargs)
    res = cycle.compute()
    
    if button_id == 'btn-export-csv-states':
        csv_content = export_states_csv(cycle)
        return dict(content=csv_content, filename=f'{cycle_type}_states.csv',
                   type='text/csv')
    
    elif button_id == 'btn-export-svg':
        if diagram_type == 'Ts':
            fig = plot_Ts_diagram(cycle)
        elif diagram_type == 'Pv':
            fig = plot_Pv_diagram(cycle)
        elif diagram_type == 'hs':
            fig = plot_hs_diagram(cycle)
        else:
            fig = plot_exergy_bar(res)
        
        # 保存到临时文件再读取
        tmp = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
        tmp.close()
        ok, msg = export_figure_svg(fig, tmp.name)
        if ok:
            with open(tmp.name, 'rb') as f:
                content = f.read()
            os.unlink(tmp.name)
            return dict(content=base64.b64encode(content).decode(),
                       filename=f'{cycle_type}_{diagram_type}.svg',
                       type='image/svg+xml', base64=True)
        else:
            return dict(content=f'导出失败: {msg}', filename='error.txt')
    
    elif button_id == 'btn-export-pdf':
        # 生成所有图
        figures = {
            'T-s图': plot_Ts_diagram(cycle),
            'P-v图': plot_Pv_diagram(cycle),
            'h-s(Mollier)图': plot_hs_diagram(cycle),
            '㶲损失分布': plot_exergy_bar(res),
        }
        tmp = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp.close()
        ok, msg = generate_pdf_report(cycle, figures, tmp.name)
        if ok:
            with open(tmp.name, 'rb') as f:
                content = f.read()
            os.unlink(tmp.name)
            return dict(content=base64.b64encode(content).decode(),
                       filename=f'{cycle_type}_report.pdf',
                       type='application/pdf', base64=True)
        else:
            return dict(content=f'PDF生成失败: {msg}', filename='error.txt')
    
    raise PreventUpdate


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("  🔥 工程热力学循环分析与热效率计算工具")
    print("  访问地址: http://127.0.0.1:8050")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=8050)
