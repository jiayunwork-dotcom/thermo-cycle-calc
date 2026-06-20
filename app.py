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
from dash import dcc, html, dash_table, Input, Output, State, callback_context, ALL, MATCH
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
    plot_exergy_bar, plot_parametric_curve, plot_2d_contour,
    plot_superimposed_Ts, plot_comparison_radar
)
from analysis import (
    parametric_sweep, multi_param_sweep, exergy_analysis_summary,
    GeneticOptimizer, MIN_QUALITY
)
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
                dcc.Download(id='download-compare-png'),
                dcc.Download(id='download-compare-csv'),
            ], style={'marginBottom': 18}),
            
            # ====== 工况管理模块 ======
            html.Div([
                html.Label('📁 工况管理 (最多8组)', 
                          style={'fontWeight': 'bold', 'display': 'block',
                                 'marginBottom': 8, 'fontSize': 13}),
                
                # 保存工况
                html.Div([
                    html.Div([
                        dcc.Input(id='case-name-input', type='text',
                                  placeholder='输入工况名称...',
                                  value='',
                                  style={'width': '65%', 'padding': 6,
                                         'borderRadius': 4, 'border': '1px solid #ccc',
                                         'fontSize': 12}),
                        html.Button('💾 保存当前工况', id='btn-save-case', n_clicks=0,
                                   style={'background': '#2980b9', 'color': 'white',
                                          'border': 'none', 'padding': '6px 10px',
                                          'borderRadius': 4, 'fontSize': 12,
                                          'cursor': 'pointer', 'width': '32%',
                                          'marginLeft': '3%'}),
                    ], style={'display': 'flex', 'marginBottom': 10}),
                    html.Div(id='case-save-status', style={'fontSize': 11,
                                                           'marginBottom': 8,
                                                           'minHeight': '16px'}),
                ]),
                
                # 工况列表
                html.Div(id='saved-cases-list',
                        style={'maxHeight': '240px', 'overflowY': 'auto',
                               'padding': 4, 'background': '#f8f9fa',
                               'borderRadius': 6, 'border': '1px solid #dee2e6',
                               'minHeight': '60px'}),
                
                # 对比操作
                html.Div([
                    html.Div([
                        html.Label('选择对比模式:',
                                  style={'fontSize': 12, 'display': 'block',
                                         'marginBottom': 4}),
                        dcc.Dropdown(
                            id='compare-mode',
                            options=[
                                {'label': '总览对比 (2~4组)', 'value': 'overview'},
                                {'label': '敏感度分析 (2组)', 'value': 'sensitivity'},
                            ],
                            value='overview',
                            clearable=False,
                            style={'fontSize': 12, 'marginBottom': 8}
                        ),
                    ]),
                    html.Button('📊 开始对比', id='btn-start-compare', n_clicks=0,
                               style={'background': '#8e44ad', 'color': 'white',
                                      'border': 'none', 'padding': '8px 16px',
                                      'borderRadius': 4, 'fontSize': 13,
                                      'cursor': 'pointer', 'width': '100%',
                                      'fontWeight': 'bold'}),
                ], style={'marginTop': 12, 'padding': 10,
                         'background': '#eaf2f8', 'borderRadius': 6,
                         'border': '1px solid #aed6f1'}),
                
                html.Div(id='compare-error-msg',
                        style={'marginTop': 8, 'fontSize': 12, 'color': '#c0392b',
                               'minHeight': '16px'}),
                
            ], style={'padding': 12, 'background': 'white',
                     'borderRadius': 8, 'border': '1px solid #dee2e6',
                     'marginBottom': 18}),
            
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
                dcc.Tab(label='🔬 工况对比', value='tab-compare',
                       style={'padding': 10, 'fontWeight': 'bold',
                              'background': 'linear-gradient(135deg, #e8daef, #d6eaf8)',
                              'color': '#6c3483'}),
                dcc.Tab(label='🎯 循环优化求解器', value='tab-optimize',
                       style={'padding': 10, 'fontWeight': 'bold',
                              'background': 'linear-gradient(135deg, #fcf3cf, #fadbd8)',
                              'color': '#935116'}),
            ], style={'marginBottom': 12}),
            
            html.Div(id='tab-content'),
            
        ], style={'flex': 1, 'padding': 16, 'background': 'white'}),
        
    ], style={'display': 'flex'}),
    
    # 隐藏的存储
    dcc.Store(id='cycle-data-store'),
    # 工况存储: list of case dicts
    dcc.Store(id='cases-store', data=[]),
    # 对比结果存储
    dcc.Store(id='compare-result-store', data=None),
    # 工况选择状态存储 (保存勾选的工况ID)
    dcc.Store(id='cases-selection-store', data=[]),
    # 优化结果存储
    dcc.Store(id='optimize-result-store', data=None),
    # 优化进度存储 (当前代数)
    dcc.Store(id='optimize-progress-store', data=None),
    # 优化运行状态
    dcc.Store(id='optimize-running-store', data=False),
    
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


# Tab内容渲染 (统一处理所有tabs)
@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value'),
               Input('cycle-data-store', 'data'),
               Input('diagram-type', 'value'),
               Input('compare-result-store', 'data'),
               Input('optimize-result-store', 'data'),
               Input('optimize-progress-store', 'data')])
def render_tab(tab, store_data, diagram_type, compare_result,
               opt_result, opt_progress):
    # ====== 优化求解器Tab (独立处理) ======
    if tab == 'tab-optimize':
        return _render_optimize_tab(opt_result, opt_progress)
    
    # ====== 工况对比Tab (独立处理) ======
    if tab == 'tab-compare':
        if not compare_result:
            return html.Div([
                html.Div([
                    html.H3('🔬 工况对比分析',
                           style={'marginTop': 0, 'color': '#6c3483'}),
                    html.Div([
                        html.P('使用说明:', style={'fontWeight': 'bold',
                                                   'color': '#2c3e50'}),
                        html.Ol([
                            html.Li('在左侧"工况管理"区域，完成计算后点击"💾 保存当前工况"将结果保存（可命名）'),
                            html.Li('在已保存工况列表中勾选2~4个工况'),
                            html.Li('选择对比模式：'),
                            html.Ul([
                                html.Li('总览对比 (2~4组): 关键指标表 + 叠加T-s图 + 雷达图'),
                                html.Li('敏感度分析 (2组): 两工况参数差异对指标的影响分析'),
                            ], style={'marginLeft': 20}),
                            html.Li('点击"📊 开始对比"按钮查看结果'),
                        ], style={'lineHeight': 1.8}),
                    ], style={'padding': 20, 'background': '#f8f9fa',
                             'borderRadius': 8, 'borderLeft': '4px solid #8e44ad',
                             'marginTop': 16}),
                ], style={'maxWidth': 800, 'margin': '0 auto'}),
            ], style={'padding': 20})
        
        mode = compare_result.get('mode', 'overview')
        cases = compare_result.get('cases', [])
        
        if mode == 'overview':
            return _render_compare_overview(cases, compare_result)
        elif mode == 'sensitivity':
            return _render_compare_sensitivity(cases, compare_result)
        else:
            return html.Div('未知对比模式')
    
    # ====== 其他Tab ======
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
        if tab == 'tab-diagram':
            return _render_diagram_tab(None, {}, diagram_type)
        elif tab == 'tab-states':
            return html.Div('无法重新构建循环数据')
        elif tab == 'tab-exergy':
            return _render_exergy_tab(None, {})
        elif tab == 'tab-parametric':
            return _render_parametric_tab(cycle_type)
        else:
            return html.Div()
    
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
# 工况对比功能 - 辅助函数
# ============================================================

MAX_CASES = 8
CASE_COLORS = ['#e74c3c', '#3498db', '#27ae60', '#f39c12',
               '#8e44ad', '#16a085', '#2c3e50', '#d35400']


def _generate_case_id():
    """生成唯一工况ID"""
    return f"case_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"


def _extract_cycle_fluid_type(cycle_type):
    """判断循环工质类型: water 或 gas"""
    water_cycles = {'rankine_basic', 'rankine_reheat', 'rankine_regen'}
    gas_cycles = {'brayton_basic', 'brayton_regen', 'otto', 'diesel'}
    if cycle_type in water_cycles:
        return 'water'
    elif cycle_type in gas_cycles:
        return 'gas'
    elif cycle_type == 'ccgt':
        return 'mixed'
    return 'water'


def _compute_exergy_eff(results):
    """计算㶲效率"""
    w_net = results.get('w_net', results.get('W_dot_total_kW', 0))
    q_in = results.get('q_in', 0)
    T0 = 298.15
    # 估计热源温度: 用锅炉出口温度或TIT
    T_h_est = None
    # 从结果中可能没有，用一个合理的高温估计
    if T_h_est is None:
        T_h_est = 1000  # K 近似
    exergy_input = q_in * (1 - T0 / T_h_est) if q_in > 0 else 0
    if exergy_input > 0:
        return w_net / exergy_input
    return 0


def _compute_compactness(cycle):
    """计算紧凑性指标: 1 / max(比体积变化)"""
    try:
        vs = [sp.v for sp in cycle.states.values() if sp.v and sp.v > 0]
        if len(vs) >= 2:
            delta_v = max(vs) / min(vs)
            return 1.0 / delta_v if delta_v > 0 else 0
    except:
        pass
    return 0.01


def _rebuild_cycle_from_case(case_data):
    """从保存的工况数据重建循环对象"""
    cycle_type = case_data['cycle_type']
    params = case_data['params']
    try:
        cycle, res = build_cycle(cycle_type, params)
        return cycle, res
    except:
        return None, None


# ============================================================
# 工况对比功能 - 回调函数
# ============================================================

# ---- 1. 保存工况 ----
@app.callback(
    [Output('cases-store', 'data'),
     Output('case-save-status', 'children'),
     Output('case-name-input', 'value')],
    [Input('btn-save-case', 'n_clicks')],
    [State('cycle-type', 'value'),
     State('case-name-input', 'value'),
     State('cases-store', 'data'),
     State('cycle-data-store', 'data')] +
    [State(f"param-{k}-{p['key']}", 'value') 
     for k in CYCLE_CONFIGS 
     for p in CYCLE_CONFIGS[k]['params']],
    prevent_initial_call=True
)
def save_case(n_clicks, cycle_type, case_name, saved_cases, store_data, *all_params):
    if n_clicks == 0 or not store_data:
        raise PreventUpdate
    
    saved_cases = saved_cases or []
    
    # 检查上限
    if len(saved_cases) >= MAX_CASES:
        return saved_cases, html.Span(f'❌ 已达最大工况数({MAX_CASES})，请先删除部分工况',
                                      style={'color': '#c0392b'}), case_name
    
    # 收集当前参数
    cfg = CYCLE_CONFIGS[cycle_type]
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
    
    # 重建循环以获取完整状态点数据
    try:
        cycle, res = build_cycle(cycle_type, param_dict)
    except Exception as e:
        return saved_cases, html.Span(f'❌ 保存失败: {str(e)[:40]}',
                                      style={'color': '#c0392b'}), case_name
    
    # 工况名称
    if not case_name or case_name.strip() == '':
        case_name = f"{cfg['name']}_{len(saved_cases)+1}"
    case_name = case_name.strip()[:30]
    
    # 提取状态点数据 (转为可序列化格式)
    states_dict = {}
    for label, sp in cycle.states.items():
        states_dict[label] = sp.to_dict()
    
    # 创建工况记录
    case_id = _generate_case_id()
    case_record = {
        'id': case_id,
        'name': case_name,
        'cycle_type': cycle_type,
        'cycle_name': cfg['name'],
        'params': param_dict,
        'results': _make_serializable(res),
        'states': states_dict,
        'fluid_type': _extract_cycle_fluid_type(cycle_type),
        'color': CASE_COLORS[len(saved_cases) % len(CASE_COLORS)],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    saved_cases.append(case_record)
    
    status = html.Span(f'✅ 已保存工况 "{case_name}" ({len(saved_cases)}/{MAX_CASES})',
                       style={'color': '#27ae60'})
    
    return saved_cases, status, ''


# ---- 2. 渲染工况列表 ----
@app.callback(
    Output('saved-cases-list', 'children'),
    [Input('cases-store', 'data'),
     Input('cases-selection-store', 'data')]
)
def render_cases_list(saved_cases, selection):
    saved_cases = saved_cases or []
    selection = selection or []
    
    if not saved_cases:
        return html.Div([
            html.Div('📭 暂无保存的工况',
                    style={'textAlign': 'center', 'padding': 20,
                           'color': '#95a5a6', 'fontSize': 12})
        ])
    
    children = []
    for i, case in enumerate(saved_cases):
        cid = case['id']
        color = case.get('color', CASE_COLORS[i % len(CASE_COLORS)])
        is_checked = cid in selection
        
        row = html.Div([
            # 颜色标识
            html.Div(style={'width': 4, 'background': color,
                           'borderRadius': 2, 'marginRight': 8}),
            # 勾选框
            dcc.Checklist(
                options=[{'label': '', 'value': cid}],
                value=[cid] if is_checked else [],
                id={'type': 'case-checkbox', 'index': cid},
                style={'display': 'inline-block', 'marginRight': 6,
                       'marginTop': 2}
            ),
            # 工况信息
            html.Div([
                html.Div([
                    html.Strong(case['name'], style={'fontSize': 12}),
                    html.Span(f" ({case['cycle_name']})",
                             style={'fontSize': 10, 'color': '#7f8c8d',
                                    'marginLeft': 4}),
                ]),
                html.Div([
                    html.Span(f"η={case.get('results',{}).get('eta',case.get('results',{}).get('eta_total',0))*100:.1f}%",
                             style={'fontSize': 10, 'color': '#2c3e50'}),
                    html.Span(f" · W={case.get('results',{}).get('w_net',case.get('results',{}).get('W_dot_total_kW',0)):.1f}",
                             style={'fontSize': 10, 'color': '#2c3e50', 'marginLeft': 6}),
                    html.Span(f" · {case.get('created_at','')[5:16]}",
                             style={'fontSize': 9, 'color': '#bdc3c7', 'marginLeft': 6}),
                ], style={'marginTop': 2}),
            ], style={'flex': 1, 'minWidth': 0}),
            # 操作按钮
            html.Div([
                # 重命名按钮
                html.Button('✏️', id={'type': 'btn-rename', 'index': cid},
                           title='重命名', n_clicks=0,
                           style={'background': 'transparent', 'border': 'none',
                                  'cursor': 'pointer', 'fontSize': 12,
                                  'padding': '2px 4px'}),
                # 删除按钮
                html.Button('🗑️', id={'type': 'btn-delete', 'index': cid},
                           title='删除', n_clicks=0,
                           style={'background': 'transparent', 'border': 'none',
                                  'cursor': 'pointer', 'fontSize': 12,
                                  'padding': '2px 4px', 'color': '#e74c3c'}),
            ], style={'display': 'flex', 'gap': 2}),
        ], style={'display': 'flex', 'alignItems': 'flex-start',
                  'padding': '6px 4px', 'borderBottom': '1px solid #ecf0f1',
                  'gap': 2})
        children.append(row)
    
    return html.Div(children)


# ---- 3. 工况选择状态同步 ----
@app.callback(
    Output('cases-selection-store', 'data'),
    [Input({'type': 'case-checkbox', 'index': dash.ALL}, 'value')],
    [State('cases-selection-store', 'data'),
     State('cases-store', 'data')],
    prevent_initial_call=False
)
def sync_case_selection(checkbox_values, current_selection, saved_cases):
    ctx = callback_context
    if not ctx.triggered:
        return current_selection or []
    
    saved_cases = saved_cases or []
    valid_ids = {c['id'] for c in saved_cases}
    new_selection = []
    
    # checkbox_values 是一个 list of list (每个checklist的value)
    for vals in checkbox_values:
        if isinstance(vals, list):
            for v in vals:
                if v in valid_ids and v not in new_selection:
                    new_selection.append(v)
    
    return new_selection


# ---- 4. 删除工况 ----
@app.callback(
    Output('cases-store', 'data', allow_duplicate=True),
    [Input({'type': 'btn-delete', 'index': dash.ALL}, 'n_clicks')],
    [State('cases-store', 'data')],
    prevent_initial_call=True
)
def delete_case(delete_clicks, saved_cases):
    ctx = callback_context
    if not ctx.triggered or not saved_cases:
        raise PreventUpdate
    
    # 找到被点击的按钮ID
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    # 解析 pattern-matching ID
    try:
        import json as _json
        id_dict = _json.loads(prop_id.split('.')[0])
        case_id = id_dict.get('index')
    except:
        raise PreventUpdate
    
    new_cases = [c for c in saved_cases if c['id'] != case_id]
    # 重新分配颜色
    for i, c in enumerate(new_cases):
        c['color'] = CASE_COLORS[i % len(CASE_COLORS)]
    return new_cases


# ---- 5. 重命名工况 (打开输入对话框, 简化实现: 用dcc.Input在弹窗, 这里简化为用confirmDialog) ----
# 简化版: 点击重命名按钮后弹出prompt对话框让用户输入
# 由于Dash没有原生prompt, 这里用一个隐藏的div+input作为重命名模态框
# 为简化代码, 我们添加一个单独的回调, 使用store传递重命名信息

# 先在layout中添加, 这里用一个简单方案: 直接用一个小输入框作为行内编辑
# 考虑到复杂性, 我们采用另一种方式: 用存储+callback处理重命名
# 这里简化: 点击重命名按钮后, 在case-name-input中填入原名, 并提示覆盖
@app.callback(
    [Output('case-name-input', 'value', allow_duplicate=True),
     Output('case-save-status', 'children', allow_duplicate=True)],
    [Input({'type': 'btn-rename', 'index': dash.ALL}, 'n_clicks')],
    [State('cases-store', 'data')],
    prevent_initial_call=True
)
def prepare_rename_case(rename_clicks, saved_cases):
    ctx = callback_context
    if not ctx.triggered or not saved_cases:
        raise PreventUpdate
    
    triggered = ctx.triggered[0]
    if float(triggered.get('value', 0)) == 0:
        raise PreventUpdate
    
    try:
        import json as _json
        prop_id = triggered['prop_id']
        id_dict = _json.loads(prop_id.split('.')[0])
        case_id = id_dict.get('index')
    except:
        raise PreventUpdate
    
    case = next((c for c in saved_cases if c['id'] == case_id), None)
    if not case:
        raise PreventUpdate
    
    # 在状态中标记需要重命名的ID (用status显示提示, 实际保存时如果名称已存在会覆盖同名)
    # 简化: 填入输入框, 用户编辑后点击保存会检测并询问(简化为直接更新)
    # 这里我们将重命名功能改为: 直接修改并保存
    # 为了支持真实重命名, 我们增加一个store: rename-target-id
    return (case['name'], 
            html.Span(f'📝 正在重命名 "{case["name"]}", 修改名称后再次点击"保存"会更新此工况 (先删除再保存)', 
                     style={'color': '#2980b9'}))


# ============================================================
# 工况对比 - 执行对比
# ============================================================

@app.callback(
    [Output('compare-result-store', 'data'),
     Output('compare-error-msg', 'children')],
    [Input('btn-start-compare', 'n_clicks')],
    [State('cases-store', 'data'),
     State('cases-selection-store', 'data'),
     State('compare-mode', 'value')],
    prevent_initial_call=True
)
def execute_compare(n_clicks, saved_cases, selection, compare_mode):
    if n_clicks == 0 or not saved_cases or not selection:
        raise PreventUpdate
    
    selected_cases = [c for c in saved_cases if c['id'] in selection]
    n_selected = len(selected_cases)
    
    if compare_mode == 'overview':
        if n_selected < 2 or n_selected > 4:
            return None, html.Span(f'❌ 总览对比需选择2-4组工况 (当前选了{n_selected}组)')
    elif compare_mode == 'sensitivity':
        if n_selected != 2:
            return None, html.Span(f'❌ 敏感度分析需恰好选择2组工况 (当前选了{n_selected}组)')
    
    # 构建对比数据
    compare_result = {
        'mode': compare_mode,
        'cases': selected_cases,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    return compare_result, ''



# ============================================================
# 对比 - 总览视图
# ============================================================

def _render_compare_overview(cases, compare_result):
    """渲染总览对比: 指标表 + 叠加T-s图 + 雷达图"""
    
    # --- 1. 总览表数据准备 ---
    metrics = [
        ('热效率 η', 'eta', '%', True),       # (名称, key, 单位, 越高越好)
        ('Carnot效率', 'eta_carnot', '%', True),
        ('净输出功', 'w_net', 'kJ/kg', True),
        ('吸热量', 'q_in', 'kJ/kg', None),
        ('放热量', 'q_out', 'kJ/kg', False),
        ('㶲效率', 'exergy_eff', '%', True),
        ('(1-放热率)', 'heat_util', '%', True),
    ]
    
    # 提取各工况的指标值
    case_metrics = []
    for case in cases:
        res = case.get('results', {})
        # 重建循环以计算一些附加指标
        cycle, _ = _rebuild_cycle_from_case(case)
        
        eta = res.get('eta', res.get('eta_total', 0)) or 0
        eta_carnot = res.get('eta_carnot', 0) or 0
        w_net = res.get('w_net', res.get('W_dot_total_kW', 0)) or 0
        q_in = res.get('q_in', 0) or 0
        q_out = res.get('q_out', 0) or 0
        
        exergy_eff = _compute_exergy_eff(res)
        heat_util = 1 - (q_out / q_in) if q_in > 0 else 0
        
        case_metrics.append({
            'case': case,
            'cycle': cycle,
            'eta': eta,
            'eta_carnot': eta_carnot,
            'w_net': w_net,
            'q_in': q_in,
            'q_out': q_out,
            'exergy_eff': exergy_eff,
            'heat_util': heat_util,
        })
    
    # 找每列最优和最差
    def _find_extremes(key, higher_better):
        vals = [cm[key] for cm in case_metrics]
        if higher_better is True:
            return max(vals), min(vals)
        elif higher_better is False:
            return min(vals), max(vals)
        else:
            return None, None
    
    # --- 2. 构建总览表 ---
    table_header = ['指标', '单位'] + [cm['case']['name'] for cm in case_metrics]
    table_rows = []
    
    for metric_name, metric_key, unit, higher_better in metrics:
        row_vals = []
        vals = [cm[metric_key] for cm in case_metrics]
        best_val, worst_val = _find_extremes(metric_key, higher_better)
        
        for i, cm in enumerate(case_metrics):
            val = cm[metric_key]
            if unit == '%':
                val_str = f'{val*100:.2f}%'
            else:
                val_str = f'{val:.2f}'
            
            # 标记颜色
            style = {}
            if higher_better is not None and len(case_metrics) >= 2:
                if abs(val - best_val) < 1e-9:
                    style = {'backgroundColor': '#d4efdf',
                             'color': '#186a3b', 'fontWeight': 'bold'}
                elif abs(val - worst_val) < 1e-9:
                    style = {'backgroundColor': '#fadbd8',
                             'color': '#922b21', 'fontWeight': 'bold'}
            
            row_vals.append(html.Td(val_str, style=style))
        
        table_rows.append(html.Tr(
            [html.Td(html.Strong(metric_name),
                    style={'backgroundColor': '#f2f3f4',
                           'padding': '8px 12px'}),
             html.Td(unit, style={'color': '#7f8c8d',
                                 'padding': '8px 12px'})] + row_vals
        ))
    
    overview_table = html.Div([
        html.H4('📋 对比总览表', style={'marginTop': 0, 'color': '#6c3483'}),
        html.Div([
            html.Span('✅ 绿色加粗 = 最优值', 
                     style={'background': '#d4efdf', 'padding': '4px 10px',
                            'borderRadius': 4, 'fontSize': 12, 'marginRight': 10}),
            html.Span('⚠️ 红色加粗 = 最差值', 
                     style={'background': '#fadbd8', 'padding': '4px 10px',
                            'borderRadius': 4, 'fontSize': 12}),
        ], style={'marginBottom': 10}),
        html.Table([
            html.Thead(html.Tr(
                [html.Th(h, style={'backgroundColor': '#8e44ad',
                                  'color': 'white', 'padding': '10px 14px',
                                  'textAlign': 'left'})
                 for h in table_header]
            )),
            html.Tbody(table_rows),
        ], style={'width': '100%', 'borderCollapse': 'collapse',
                  'fontSize': 13, 'marginBottom': 20,
                  'background': 'white'}),
    ], style={'padding': 16, 'background': '#faf5fc',
              'borderRadius': 8, 'border': '1px solid #d7bde2',
              'marginBottom': 20})
    
    # --- 3. 叠加T-s图 ---
    cycles_data = []
    for cm in case_metrics:
        case = cm['case']
        cycle = cm['cycle']
        if cycle:
            cycles_data.append({
                'name': case['name'],
                'cycle': cycle,
                'fluid_type': case.get('fluid_type', 'water'),
                'color': case.get('color', '#3498db'),
            })
    
    ts_fig = plot_superimposed_Ts(
        cycles_data, 
        title=f'叠加T-s图 - {len(cases)}组工况对比'
    ) if cycles_data else go.Figure()
    
    ts_plot_div = html.Div([
        html.H4('🗺️ 叠加T-s图 (多工况循环路径)',
               style={'marginTop': 0, 'color': '#6c3483'}),
        dcc.Graph(id='compare-ts-figure', figure=ts_fig,
                 style={'height': '550px'})
    ], style={'padding': 16, 'background': '#f8f9fa',
              'borderRadius': 8, 'border': '1px solid #dee2e6',
              'marginBottom': 20})
    
    # --- 4. 雷达图 ---
    radar_cases = []
    for cm in case_metrics:
        case = cm['case']
        cycle = cm['cycle']
        compactness = _compute_compactness(cycle) if cycle else 0.01
        q_in = cm['q_in']
        heat_rej_ratio = cm['q_out'] / q_in if q_in > 0 else 0
        
        radar_cases.append({
            'name': case['name'],
            'eta': cm['eta'],
            'w_net': cm['w_net'],
            'exergy_eff': cm['exergy_eff'],
            'heat_rejection_ratio': heat_rej_ratio,
            'compactness': compactness,
            'color': case.get('color', '#3498db'),
        })
    
    radar_fig = plot_comparison_radar(
        radar_cases,
        title='综合性能雷达图 (五维度对比)'
    )
    
    radar_plot_div = html.Div([
        html.H4('🎯 综合性能雷达图',
               style={'marginTop': 0, 'color': '#6c3483'}),
        html.Div([
            html.Span('维度说明:', style={'fontWeight': 'bold', 'fontSize': 12}),
            html.Span(' 热效率 · 净功 · 㶲效率 · 热量利用率 · 紧凑性',
                     style={'fontSize': 12, 'color': '#566573', 'marginLeft': 6}),
        ], style={'marginBottom': 10}),
        dcc.Graph(id='compare-radar-figure', figure=radar_fig,
                 style={'height': '600px'})
    ], style={'padding': 16, 'background': '#fdfefe',
              'borderRadius': 8, 'border': '1px solid #d1f2eb',
              'marginBottom': 20})
    
    # --- 5. 导出按钮 ---
    export_div = html.Div([
        html.H4('📤 导出对比结果', style={'marginTop': 0, 'color': '#6c3483'}),
        html.Div([
            html.Button('🖼️ 导出为PNG截图 (图表+表格)',
                       id='btn-export-compare-png', n_clicks=0,
                       style={'background': '#2980b9', 'color': 'white',
                              'border': 'none', 'padding': '10px 20px',
                              'borderRadius': 6, 'fontSize': 14,
                              'cursor': 'pointer', 'marginRight': 12,
                              'fontWeight': 'bold'}),
            html.Button('📊 导出总览表为CSV',
                       id='btn-export-compare-csv', n_clicks=0,
                       style={'background': '#27ae60', 'color': 'white',
                              'border': 'none', 'padding': '10px 20px',
                              'borderRadius': 6, 'fontSize': 14,
                              'cursor': 'pointer',
                              'fontWeight': 'bold'}),
        ]),
        html.Div(f'生成时间: {compare_result.get("generated_at", "")}',
                style={'marginTop': 10, 'fontSize': 11, 'color': '#95a5a6'}),
    ], style={'padding': 16, 'background': '#fef9e7',
              'borderRadius': 8, 'border': '1px solid #f9e79f',
              'marginBottom': 20})
    
    return html.Div([
        html.H3('🔬 工况对比结果 - 总览模式',
               style={'marginTop': 0, 'color': '#6c3483',
                      'borderBottom': '2px solid #8e44ad',
                      'paddingBottom': 10}),
        overview_table,
        ts_plot_div,
        radar_plot_div,
        export_div,
    ], style={'padding': 4})


# ============================================================
# 对比 - 敏感度分析
# ============================================================

def _render_compare_sensitivity(cases, compare_result):
    """两工况敏感度对比分析"""
    if len(cases) != 2:
        return html.Div('需要恰好两组工况进行敏感度分析')
    
    case_a, case_b = cases
    
    # --- 重建循环以获取完整数据 ---
    cycle_a, res_a = _rebuild_cycle_from_case(case_a)
    cycle_b, res_b = _rebuild_cycle_from_case(case_b)
    
    params_a = case_a.get('params', {})
    params_b = case_b.get('params', {})
    
    # --- 找出共同参数 ---
    common_params = sorted(set(params_a.keys()) & set(params_b.keys()))
    
    # 找出变化的参数
    changed_params = []
    for p in common_params:
        va, vb = params_a.get(p), params_b.get(p)
        if va != vb:
            # 获取参数label和单位
            cfg_a = CYCLE_CONFIGS.get(case_a['cycle_type'], {})
            param_info = next((x for x in cfg_a.get('params', []) if x['key'] == p), None)
            label = param_info['label'] if param_info else p
            unit = param_info['unit'] if param_info else ''
            changed_params.append({
                'key': p,
                'label': label,
                'unit': unit,
                'value_a': va,
                'value_b': vb,
                'delta': vb - va if (isinstance(va, (int, float)) and isinstance(vb, (int, float))) else None,
            })
    
    # --- 计算输出指标变化 ---
    def _get_metric(res, key):
        if key == 'eta':
            return res.get('eta', res.get('eta_total', 0)) or 0
        elif key == 'eta_carnot':
            return res.get('eta_carnot', 0) or 0
        elif key == 'w_net':
            return res.get('w_net', res.get('W_dot_total_kW', 0)) or 0
        elif key == 'q_in':
            return res.get('q_in', 0) or 0
        elif key == 'q_out':
            return res.get('q_out', 0) or 0
        elif key == 'exergy_eff':
            return _compute_exergy_eff(res)
        return 0
    
    output_metrics = [
        ('热效率 η', 'eta', '%'),
        ('Carnot效率', 'eta_carnot', '%'),
        ('净输出功', 'w_net', 'kJ/kg'),
        ('吸热量', 'q_in', 'kJ/kg'),
        ('放热量', 'q_out', 'kJ/kg'),
        ('㶲效率', 'exergy_eff', '%'),
    ]
    
    sensitivity_rows = []
    for metric_name, metric_key, unit in output_metrics:
        va = _get_metric(res_a, metric_key)
        vb = _get_metric(res_b, metric_key)
        delta = vb - va
        
        if unit == '%':
            va_str = f'{va*100:.3f}%'
            vb_str = f'{vb*100:.3f}%'
            delta_str = f'{delta*100:+.3f} pp'  # percentage points
            pct_str = f'{(delta/va*100):+.2f}%' if va != 0 else 'N/A'
        else:
            va_str = f'{va:.3f}'
            vb_str = f'{vb:.3f}'
            delta_str = f'{delta:+.3f}'
            pct_str = f'{(delta/va*100):+.2f}%' if va != 0 else 'N/A'
        
        # 颜色标记
        if metric_key in ['eta', 'eta_carnot', 'w_net', 'exergy_eff']:
            # 越高越好
            if delta > 1e-9:
                delta_color = '#27ae60'
            elif delta < -1e-9:
                delta_color = '#e74c3c'
            else:
                delta_color = '#7f8c8d'
        elif metric_key == 'q_out':
            # 越低越好
            if delta < -1e-9:
                delta_color = '#27ae60'
            elif delta > 1e-9:
                delta_color = '#e74c3c'
            else:
                delta_color = '#7f8c8d'
        else:
            delta_color = '#2c3e50'
        
        sensitivity_rows.append(html.Tr([
            html.Td(html.Strong(metric_name),
                   style={'padding': '8px 12px', 'backgroundColor': '#f2f3f4'}),
            html.Td(unit, style={'padding': '8px 12px', 'color': '#7f8c8d'}),
            html.Td(va_str, style={'padding': '8px 12px', 'textAlign': 'right',
                                  'backgroundColor': '#fdf2e9'}),
            html.Td(vb_str, style={'padding': '8px 12px', 'textAlign': 'right',
                                  'backgroundColor': '#eaf2f8'}),
            html.Td(delta_str, 
                   style={'padding': '8px 12px', 'textAlign': 'right',
                          'fontWeight': 'bold', 'color': delta_color}),
            html.Td(pct_str,
                   style={'padding': '8px 12px', 'textAlign': 'right',
                          'fontWeight': 'bold', 'color': delta_color}),
        ]))
    
    # --- 变化参数表 ---
    changed_rows = []
    for cp in changed_params:
        delta = cp['delta']
        if delta is not None:
            delta_str = f'{delta:+.4g} {cp["unit"]}'.strip()
            va_str = f'{cp["value_a"]:g} {cp["unit"]}'.strip()
            vb_str = f'{cp["value_b"]:g} {cp["unit"]}'.strip()
            pct_str = f'{(delta/cp["value_a"]*100):+.2f}%' if cp['value_a'] != 0 else 'N/A'
        else:
            delta_str = f'{cp["value_a"]} → {cp["value_b"]}'
            va_str = str(cp['value_a'])
            vb_str = str(cp['value_b'])
            pct_str = '-'
        
        changed_rows.append(html.Tr([
            html.Td(html.Strong(cp['label']),
                   style={'padding': '8px 12px', 'backgroundColor': '#f2f3f4'}),
            html.Td(cp['unit'], style={'padding': '8px 12px', 'color': '#7f8c8d'}),
            html.Td(va_str, style={'padding': '8px 12px', 'textAlign': 'right',
                                  'backgroundColor': '#fdf2e9'}),
            html.Td(vb_str, style={'padding': '8px 12px', 'textAlign': 'right',
                                  'backgroundColor': '#eaf2f8'}),
            html.Td(delta_str, style={'padding': '8px 12px', 'textAlign': 'right',
                                     'fontWeight': 'bold', 'color': '#6c3483'}),
            html.Td(pct_str, style={'padding': '8px 12px', 'textAlign': 'right',
                                   'fontWeight': 'bold', 'color': '#6c3483'}),
        ]))
    
    if not changed_rows:
        changed_rows.append(html.Tr([
            html.Td(html.Span('两组工况参数完全相同,无可对比差异',
                             style={'color': '#e67e22'}),
                   colSpan=6, style={'padding': '20px', 'textAlign': 'center'})
        ]))
    
    # --- 构造卡片显示工况信息 ---
    def _make_case_card(case, color, label_text):
        return html.Div([
            html.Div(label_text, 
                    style={'fontSize': 11, 'color': 'white',
                           'backgroundColor': color, 'padding': '4px 10px',
                           'borderRadius': '4px 4px 0 0',
                           'fontWeight': 'bold'}),
            html.Div([
                html.Strong(case['name'], style={'fontSize': 15}),
                html.Div(f"{case['cycle_name']} · {case.get('created_at','')[:10]}",
                        style={'fontSize': 11, 'color': '#7f8c8d',
                               'marginTop': 2}),
            ], style={'padding': 12, 'background': 'white',
                     'border': f'2px solid {color}',
                     'borderTop': 'none',
                     'borderRadius': '0 0 8px 8px'}),
        ], style={'width': '48%'})
    
    cards_row = html.Div([
        _make_case_card(case_a, case_a.get('color', '#e74c3c'), '工况 A (基准)'),
        html.Div('→', style={'fontSize': 30, 'alignSelf': 'center',
                             'color': '#8e44ad', 'fontWeight': 'bold'}),
        _make_case_card(case_b, case_b.get('color', '#3498db'), '工况 B (对比)'),
    ], style={'display': 'flex', 'justifyContent': 'space-between',
              'alignItems': 'stretch', 'marginBottom': 20})
    
    # --- 最终组合 ---
    return html.Div([
        html.H3('🔬 工况对比结果 - 敏感度分析模式',
               style={'marginTop': 0, 'color': '#6c3483',
                      'borderBottom': '2px solid #8e44ad',
                      'paddingBottom': 10}),
        html.Div([
            html.Span('💡 分析两工况间参数变化对各输出指标的影响',
                     style={'background': '#ebf5fb', 'padding': '8px 12px',
                            'borderRadius': 6, 'fontSize': 12, 'color': '#2874a6'}),
        ], style={'marginBottom': 16}),
        
        cards_row,
        
        # 变化参数表
        html.Div([
            html.H4('🔧 变化的输入参数',
                   style={'marginTop': 0, 'color': '#6c3483'}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th('参数名', style={'background': '#d6baf3',
                                            'color': '#4a235a',
                                            'padding': '10px 12px'}),
                    html.Th('单位', style={'background': '#d6baf3',
                                          'color': '#4a235a',
                                          'padding': '10px 12px'}),
                    html.Th(f'A: {case_a["name"]}',
                           style={'background': case_a.get('color', '#e74c3c'),
                                  'color': 'white', 'padding': '10px 12px'}),
                    html.Th(f'B: {case_b["name"]}',
                           style={'background': case_b.get('color', '#3498db'),
                                  'color': 'white', 'padding': '10px 12px'}),
                    html.Th('变化量',
                           style={'background': '#af7ac5', 'color': 'white',
                                  'padding': '10px 12px'}),
                    html.Th('变化率',
                           style={'background': '#af7ac5', 'color': 'white',
                                  'padding': '10px 12px'}),
                ])),
                html.Tbody(changed_rows),
            ], style={'width': '100%', 'borderCollapse': 'collapse',
                      'background': 'white', 'marginBottom': 24}),
        ], style={'padding': 16, 'background': '#f5eef8',
                  'borderRadius': 8, 'border': '1px solid #d7bde2',
                  'marginBottom': 20}),
        
        # 输出指标变化表
        html.Div([
            html.H4('📊 输出指标变化分析',
                   style={'marginTop': 0, 'color': '#6c3483'}),
            html.Div([
                html.Span('✅ 绿色 = 变好', style={'background': '#d4efdf',
                             'padding': '4px 10px', 'borderRadius': 4,
                             'fontSize': 12, 'marginRight': 10}),
                html.Span('❌ 红色 = 变差', style={'background': '#fadbd8',
                             'padding': '4px 10px', 'borderRadius': 4,
                             'fontSize': 12}),
                html.Span(' (放热量越低越好)',
                         style={'fontSize': 11, 'color': '#7f8c8d',
                                'marginLeft': 10}),
            ], style={'marginBottom': 10}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th('输出指标', style={'background': '#8e44ad',
                                              'color': 'white',
                                              'padding': '10px 12px'}),
                    html.Th('单位', style={'background': '#8e44ad',
                                          'color': 'white',
                                          'padding': '10px 12px'}),
                    html.Th(f'A: {case_a["name"]}',
                           style={'background': case_a.get('color', '#e74c3c'),
                                  'color': 'white', 'padding': '10px 12px'}),
                    html.Th(f'B: {case_b["name"]}',
                           style={'background': case_b.get('color', '#3498db'),
                                  'color': 'white', 'padding': '10px 12px'}),
                    html.Th('变化量',
                           style={'background': '#6c3483', 'color': 'white',
                                  'padding': '10px 12px'}),
                    html.Th('变化率',
                           style={'background': '#6c3483', 'color': 'white',
                                  'padding': '10px 12px'}),
                ])),
                html.Tbody(sensitivity_rows),
            ], style={'width': '100%', 'borderCollapse': 'collapse',
                      'background': 'white'}),
        ], style={'padding': 16, 'background': '#eaf2f8',
                  'borderRadius': 8, 'border': '1px solid #aed6f1',
                  'marginBottom': 20}),
        
        # 导出按钮
        html.Div([
            html.Button('📊 导出敏感度表为CSV',
                       id='btn-export-sensitivity-csv', n_clicks=0,
                       style={'background': '#27ae60', 'color': 'white',
                              'border': 'none', 'padding': '10px 20px',
                              'borderRadius': 6, 'fontSize': 14,
                              'cursor': 'pointer', 'fontWeight': 'bold'}),
            html.Div(f'生成时间: {compare_result.get("generated_at", "")}',
                    style={'marginTop': 10, 'fontSize': 11, 'color': '#95a5a6'}),
        ], style={'padding': 16, 'background': '#fef9e7',
                  'borderRadius': 8, 'border': '1px solid #f9e79f'}),
        
    ], style={'padding': 4})


# ============================================================
# 工况对比 - 导出功能
# ============================================================

@app.callback(
    Output('download-compare-csv', 'data'),
    [Input('btn-export-compare-csv', 'n_clicks'),
     Input('btn-export-sensitivity-csv', 'n_clicks')],
    [State('compare-result-store', 'data')],
    prevent_initial_call=True
)
def export_compare_csv(n_compare_csv, n_sens_csv, compare_result):
    ctx = callback_context
    if not ctx.triggered or not compare_result:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    cases = compare_result.get('cases', [])
    mode = compare_result.get('mode', 'overview')
    
    # 提取工况指标
    def _extract_metrics(case):
        res = case.get('results', {})
        cycle, _ = _rebuild_cycle_from_case(case)
        return {
            'case_name': case['name'],
            'cycle_type': case.get('cycle_name', ''),
            'eta': res.get('eta', res.get('eta_total', 0)) or 0,
            'eta_carnot': res.get('eta_carnot', 0) or 0,
            'w_net': res.get('w_net', res.get('W_dot_total_kW', 0)) or 0,
            'q_in': res.get('q_in', 0) or 0,
            'q_out': res.get('q_out', 0) or 0,
            'exergy_eff': _compute_exergy_eff(res),
            'heat_util': 1 - (res.get('q_out',0)/res.get('q_in',1)) if res.get('q_in',0) > 0 else 0,
            'compactness': _compute_compactness(cycle) if cycle else 0,
        }
    
    if button_id == 'btn-export-compare-csv' and mode == 'overview':
        rows = []
        header = ['指标', '单位'] + [c['name'] for c in cases]
        metrics_def = [
            ('热效率 η', 'eta', '%'),
            ('Carnot效率', 'eta_carnot', '%'),
            ('净输出功', 'w_net', 'kJ/kg'),
            ('吸热量', 'q_in', 'kJ/kg'),
            ('放热量', 'q_out', 'kJ/kg'),
            ('㶲效率', 'exergy_eff', '%'),
            ('热量利用率(1-q_out/q_in)', 'heat_util', '%'),
            ('紧凑性', 'compactness', ''),
        ]
        for mname, mkey, unit in metrics_def:
            row = [mname, unit]
            for case in cases:
                m = _extract_metrics(case)
                v = m.get(mkey, 0)
                if unit == '%':
                    row.append(f'{v*100:.4f}%')
                else:
                    row.append(f'{v:.4f}')
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=header)
        csv_str = df.to_csv(index=False, encoding='utf-8-sig')
        
        return dict(content=csv_str, filename=f'工况对比总览_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                   type='text/csv')
    
    elif button_id == 'btn-export-sensitivity-csv' and mode == 'sensitivity' and len(cases) == 2:
        case_a, case_b = cases
        params_a = case_a.get('params', {})
        params_b = case_b.get('params', {})
        common_params = sorted(set(params_a.keys()) & set(params_b.keys()))
        
        # 参数变化表
        param_rows = []
        param_rows.append(['=== 输入参数变化 ===', '', '', '', '', ''])
        param_rows.append(['参数名', '单位', f'A: {case_a["name"]}', f'B: {case_b["name"]}', '变化量', '变化率(%)'])
        for p in common_params:
            va, vb = params_a.get(p), params_b.get(p)
            cfg_a = CYCLE_CONFIGS.get(case_a['cycle_type'], {})
            pinfo = next((x for x in cfg_a.get('params', []) if x['key'] == p), None)
            label = pinfo['label'] if pinfo else p
            unit = pinfo['unit'] if pinfo else ''
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = vb - va
                pct = (delta / va * 100) if va != 0 else None
                param_rows.append([label, unit, f'{va:g}', f'{vb:g}',
                                   f'{delta:+.4g}', f'{pct:+.2f}' if pct is not None else 'N/A'])
            else:
                param_rows.append([label, unit, str(va), str(vb), '-', '-'])
        
        # 指标变化表
        param_rows.append([])
        param_rows.append(['=== 输出指标变化 ===', '', '', '', '', ''])
        param_rows.append(['指标', '单位', f'A: {case_a["name"]}', f'B: {case_b["name"]}', '变化量', '变化率(%)'])
        
        ma, mb = _extract_metrics(case_a), _extract_metrics(case_b)
        out_defs = [
            ('热效率 η', 'eta', '%'),
            ('Carnot效率', 'eta_carnot', '%'),
            ('净输出功', 'w_net', 'kJ/kg'),
            ('吸热量', 'q_in', 'kJ/kg'),
            ('放热量', 'q_out', 'kJ/kg'),
            ('㶲效率', 'exergy_eff', '%'),
        ]
        for mname, mkey, unit in out_defs:
            va, vb = ma.get(mkey,0), mb.get(mkey,0)
            delta = vb - va
            if unit == '%':
                pct = (delta / va * 100) if va != 0 else None
                param_rows.append([mname, unit, f'{va*100:.4f}%', f'{vb*100:.4f}%',
                                   f'{delta*100:+.4f} pp', f'{pct:+.2f}' if pct is not None else 'N/A'])
            else:
                pct = (delta / va * 100) if va != 0 else None
                param_rows.append([mname, unit, f'{va:.4f}', f'{vb:.4f}',
                                   f'{delta:+.4f}', f'{pct:+.2f}' if pct is not None else 'N/A'])
        
        df = pd.DataFrame(param_rows)
        csv_str = df.to_csv(index=False, header=False, encoding='utf-8-sig')
        
        return dict(content=csv_str, filename=f'敏感度分析_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                   type='text/csv')
    
    raise PreventUpdate


@app.callback(
    Output('download-compare-png', 'data'),
    [Input('btn-export-compare-png', 'n_clicks')],
    [State('compare-result-store', 'data')],
    prevent_initial_call=True
)
def export_compare_png(n_clicks, compare_result):
    if n_clicks == 0 or not compare_result:
        raise PreventUpdate
    
    mode = compare_result.get('mode', 'overview')
    cases = compare_result.get('cases', [])
    
    if mode != 'overview':
        # 敏感度模式也可以导出两张图
        pass
    
    # 重建工况
    case_metrics = []
    for case in cases:
        cycle, res = _rebuild_cycle_from_case(case)
        q_in = res.get('q_in', 0) or 0
        q_out = res.get('q_out', 0) or 0
        case_metrics.append({
            'case': case,
            'cycle': cycle,
            'eta': res.get('eta', res.get('eta_total', 0)) or 0,
            'w_net': res.get('w_net', res.get('W_dot_total_kW', 0)) or 0,
            'exergy_eff': _compute_exergy_eff(res),
            'heat_rejection_ratio': q_out / q_in if q_in > 0 else 0,
            'compactness': _compute_compactness(cycle) if cycle else 0.01,
        })
    
    # 生成T-s叠加图
    cycles_data = []
    for cm in case_metrics:
        case = cm['case']
        cycle = cm['cycle']
        if cycle:
            cycles_data.append({
                'name': case['name'],
                'cycle': cycle,
                'fluid_type': case.get('fluid_type', 'water'),
                'color': case.get('color', '#3498db'),
            })
    
    fig_ts = plot_superimposed_Ts(cycles_data, title='叠加T-s图 - 工况对比')
    
    # 生成雷达图
    radar_cases = []
    for cm in case_metrics:
        case = cm['case']
        radar_cases.append({
            'name': case['name'],
            'eta': cm['eta'],
            'w_net': cm['w_net'],
            'exergy_eff': cm['exergy_eff'],
            'heat_rejection_ratio': cm['heat_rejection_ratio'],
            'compactness': cm['compactness'],
            'color': case.get('color', '#3498db'),
        })
    fig_radar = plot_comparison_radar(radar_cases, title='综合性能雷达图')
    
    # 使用subplots将两个图合在一起导出
    from plotly.subplots import make_subplots
    
    # 组合成一个大图 (上下布局)
    combined_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('叠加T-s图 (多工况循环路径对比)', '综合性能雷达图'),
        vertical_spacing=0.08,
        specs=[[{'type': 'xy'}], [{'type': 'polar'}]]
    )
    
    # 添加T-s图 traces
    for trace in fig_ts.data:
        combined_fig.add_trace(trace, row=1, col=1)
    combined_fig.update_xaxes(title_text='比熵 s [kJ/(kg·K)]', row=1, col=1)
    combined_fig.update_yaxes(title_text='温度 T [°C]', row=1, col=1)
    
    # 添加雷达图 traces
    for trace in fig_radar.data:
        combined_fig.add_trace(trace, row=2, col=1)
    combined_fig.update_layout(
        polar=fig_radar.layout.polar,
        polar_domain=dict(y=[0, 0.45])
    )
    
    combined_fig.update_layout(
        height=1100,
        width=1000,
        title=dict(text=f'工况对比分析报告 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                  font=dict(size=18, color='#6c3483'),
                  x=0.5),
        showlegend=True,
        legend=dict(x=1.02, y=0.5, xanchor='left'),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    
    # 导出为PNG
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    ok, msg = export_figure_png(combined_fig, tmp.name, scale=2)
    if ok:
        with open(tmp.name, 'rb') as f:
            content = f.read()
        os.unlink(tmp.name)
        return dict(content=base64.b64encode(content).decode(),
                   filename=f'工况对比_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
                   type='image/png', base64=True)
    else:
        return dict(content=f'PNG导出失败: {msg}', filename='export_error.txt')


# ============================================================
# 循环优化求解器 - 辅助函数
# ============================================================

def _render_optimize_tab(opt_result, opt_progress):
    """渲染优化求解器Tab"""
    return html.Div([
        html.H3('🎯 循环优化求解器 (遗传算法)',
                style={'marginTop': 0, 'color': '#935116',
                       'borderBottom': '2px solid #e67e22',
                       'paddingBottom': 10}),
        
        html.Div([
            html.Span('使用说明:', style={'fontWeight': 'bold', 'color': '#2c3e50'}),
            html.Ol([
                html.Li('选择循环类型和优化目标（最大化热效率或净功）'),
                html.Li('勾选参与优化的参数，设置其上下界（未勾选的参数将使用左侧面板当前值固定不变）'),
                html.Li('可调整遗传算法参数（种群大小/代数/交叉率/变异率），使用默认值也可'),
                html.Li('两个固定约束条件: 涡轮出口干度≥0.88、状态点温度不超过工质上限。不满足约束的解将直接淘汰。'),
                html.Li('点击"开始优化"，等待进度完成后查看最优结果'),
            ], style={'lineHeight': 1.8, 'marginTop': 6, 'marginBottom': 0}),
        ], style={'padding': 16, 'background': '#fef9e7',
                  'borderRadius': 8, 'borderLeft': '4px solid #e67e22',
                  'marginTop': 16, 'marginBottom': 20}),
        
        # ===== 上半部分: 左侧参数配置 + 右侧进度/控制 =====
        html.Div([
            # ---- 左侧: 参数配置 ----
            html.Div([
                # 1. 循环类型 + 优化目标
                html.Div([
                    html.H4('1️⃣ 基本设置', style={'marginTop': 0, 'color': '#935116'}),
                    html.Div([
                        html.Label('循环类型:', style={'fontWeight': 'bold',
                                                         'display': 'block',
                                                         'marginBottom': 6}),
                        dcc.Dropdown(
                            id='opt-cycle-type',
                            options=[{'label': v['name'], 'value': k}
                                     for k, v in CYCLE_CONFIGS.items()],
                            value='rankine_basic',
                            clearable=False,
                            style={'fontSize': 13, 'marginBottom': 14}
                        ),
                    ]),
                    html.Div([
                        html.Label('优化目标:', style={'fontWeight': 'bold',
                                                         'display': 'block',
                                                         'marginBottom': 6}),
                        dcc.RadioItems(
                            id='opt-objective',
                            options=[
                                {'label': ' 最大化热效率 η', 'value': 'eta'},
                                {'label': ' 最大化净输出功 W_net', 'value': 'w_net'},
                            ],
                            value='eta',
                            labelStyle={'display': 'block',
                                        'marginBottom': 6,
                                        'fontSize': 13}
                        ),
                    ]),
                ], style={'padding': 14, 'background': '#fdfefe',
                          'border': '1px solid #d5dbdb',
                          'borderRadius': 8, 'marginBottom': 16}),
                
                # 2. 参数上下界设置 (动态根据循环类型渲染)
                html.Div([
                    html.H4('2️⃣ 参数优化范围',
                            style={'marginTop': 0, 'color': '#935116',
                                   'marginBottom': 10}),
                    html.Div(id='opt-param-configs',
                             style={'maxHeight': '320px', 'overflowY': 'auto',
                                    'paddingRight': 8}),
                ], style={'padding': 14, 'background': '#fdfefe',
                          'border': '1px solid #d5dbdb',
                          'borderRadius': 8, 'marginBottom': 16}),
                
                # 3. 遗传算法参数
                html.Div([
                    html.H4('3️⃣ 遗传算法参数',
                            style={'marginTop': 0, 'color': '#935116',
                                   'marginBottom': 10}),
                    html.Div([
                        html.Div([
                            html.Label('种群大小:', style={'fontSize': 12,
                                                             'display': 'block',
                                                             'marginBottom': 3}),
                            dcc.Input(id='opt-pop-size', type='number',
                                      min=10, max=500, step=10, value=50,
                                      style={'width': '100%', 'padding': 5,
                                             'fontSize': 12}),
                        ], style={'width': '48%', 'display': 'inline-block',
                                  'marginRight': '4%', 'marginBottom': 10}),
                        html.Div([
                            html.Label('进化代数:', style={'fontSize': 12,
                                                             'display': 'block',
                                                             'marginBottom': 3}),
                            dcc.Input(id='opt-n-gen', type='number',
                                      min=5, max=500, step=5, value=80,
                                      style={'width': '100%', 'padding': 5,
                                             'fontSize': 12}),
                        ], style={'width': '48%', 'display': 'inline-block',
                                  'marginBottom': 10}),
                        html.Div([
                            html.Label('交叉率:', style={'fontSize': 12,
                                                           'display': 'block',
                                                           'marginBottom': 3}),
                            dcc.Input(id='opt-cx-rate', type='number',
                                      min=0.0, max=1.0, step=0.05, value=0.8,
                                      style={'width': '100%', 'padding': 5,
                                             'fontSize': 12}),
                        ], style={'width': '48%', 'display': 'inline-block',
                                  'marginRight': '4%'}),
                        html.Div([
                            html.Label('变异率:', style={'fontSize': 12,
                                                           'display': 'block',
                                                           'marginBottom': 3}),
                            dcc.Input(id='opt-mut-rate', type='number',
                                      min=0.0, max=1.0, step=0.01, value=0.1,
                                      style={'width': '100%', 'padding': 5,
                                             'fontSize': 12}),
                        ], style={'width': '48%', 'display': 'inline-block'}),
                    ]),
                    html.Div([
                        html.Small('💡 提示: 参数越多/种群越大/代数越多，计算时间越长。',
                                   style={'color': '#7f8c8d', 'fontSize': 11}),
                    ], style={'marginTop': 8}),
                ], style={'padding': 14, 'background': '#fdfefe',
                          'border': '1px solid #d5dbdb',
                          'borderRadius': 8, 'marginBottom': 16}),
                
                # 4. 开始按钮
                html.Div([
                    html.Button('🚀 开始优化', id='btn-run-optimize', n_clicks=0,
                               style={'background': '#e67e22', 'color': 'white',
                                      'border': 'none', 'padding': '12px 24px',
                                      'borderRadius': 6, 'fontSize': 15,
                                      'fontWeight': 'bold', 'cursor': 'pointer',
                                      'width': '100%'}),
                ]),
            ], style={'width': '420px', 'flexShrink': 0,
                      'paddingRight': 20}),
            
            # ---- 右侧: 进度 + 结果预览 ----
            html.Div([
                # 进度显示
                html.Div(id='opt-progress-panel',
                         children=_render_opt_progress(opt_progress,
                                                        opt_result is not None),
                         style={'padding': 16, 'background': '#eaf2f8',
                                'border': '1px solid #aed6f1',
                                'borderRadius': 8, 'marginBottom': 16,
                                'minHeight': '120px'}),
                
                # 保存为工况按钮 (静态放置, 始终存在)
                html.Div([
                    html.Div([
                        html.Button('💾 保存最优解为工况',
                                   id='btn-save-opt-case', n_clicks=0,
                                   disabled=(opt_result is None or 'error' in opt_result),
                                   style={'background': '#2980b9' if opt_result and 'error' not in opt_result else '#95a5a6',
                                          'color': 'white',
                                          'border': 'none', 'padding': '10px 20px',
                                          'borderRadius': 6, 'fontSize': 13,
                                          'fontWeight': 'bold', 'cursor': 'pointer',
                                          'whiteSpace': 'nowrap', 'width': '100%'}),
                        html.Div(id='opt-save-status',
                                style={'fontSize': 11, 'marginTop': 6,
                                       'minHeight': '14px', 'textAlign': 'center'}),
                    ]),
                ], style={'padding': 14, 'background': '#ebf5fb',
                          'border': '1px solid #aed6f1',
                          'borderRadius': 8, 'marginBottom': 16}),
                
                # 约束说明
                html.Div([
                    html.Strong('📌 固定约束条件 (不满足将淘汰):',
                               style={'color': '#922b21', 'display': 'block',
                                      'marginBottom': 6}),
                    html.Ul([
                        html.Li(f'涡轮出口干度 x ≥ {MIN_QUALITY}'
                                ' (对水蒸气循环有效)'),
                        html.Li('所有状态点温度 ≤ 工质上限'
                                ' (水蒸气650°C / 空气1500°C)'),
                    ], style={'fontSize': 12, 'color': '#616a6b',
                              'margin': 0, 'paddingLeft': 22}),
                ], style={'padding': 12, 'background': '#fdedec',
                          'border': '1px solid #f5b7b1',
                          'borderRadius': 8, 'fontSize': 12}),
            ], style={'flex': 1}),
        ], style={'display': 'flex', 'marginBottom': 20}),
        
        # ===== 下半部分: 优化结果展示 =====
        html.Div(id='opt-results-section',
                 children=_render_opt_results(opt_result) if opt_result else html.Div(
                     [html.Div('📊 优化完成后将在此处显示结果',
                              style={'textAlign': 'center',
                                     'padding': 40,
                                     'color': '#95a5a6',
                                     'background': '#f8f9fa',
                                     'borderRadius': 8})])),
    ], style={'padding': 4})


def _render_opt_progress(opt_progress, finished=False):
    """渲染优化进度面板"""
    if not opt_progress and not finished:
        return html.Div([
            html.Strong('⏳ 等待开始优化...',
                       style={'color': '#2c3e50', 'fontSize': 14}),
            html.Div('设置好参数后点击"开始优化"按钮',
                    style={'color': '#7f8c8d', 'fontSize': 12,
                           'marginTop': 6}),
        ])
    
    if finished:
        gen = opt_progress.get('generation', 0) if opt_progress else '?'
        total = opt_progress.get('total', '?') if opt_progress else '?'
        best = opt_progress.get('best_fitness', 0) if opt_progress else 0
        obj_type = opt_progress.get('objective', 'eta') if opt_progress else 'eta'
        
        best_str = (f'{best*100:.3f}%' if obj_type == 'eta'
                    else f'{best:.3f} kJ/kg')
        return html.Div([
            html.Div([
                html.Span('✅ 优化完成!',
                         style={'color': '#27ae60', 'fontSize': 16,
                                'fontWeight': 'bold'}),
            ], style={'marginBottom': 6}),
            html.Div([
                html.Span(f'共运行 {gen}/{total} 代',
                         style={'color': '#2c3e50', 'fontSize': 13}),
                html.Span(f'  |  最佳目标值: {best_str}',
                         style={'color': '#e67e22', 'fontSize': 13,
                                'fontWeight': 'bold', 'marginLeft': 12}),
            ]),
            # 进度条 (满格)
            html.Div([
                html.Div(style={'width': '100%',
                                'height': '20px',
                                'background': '#27ae60',
                                'borderRadius': 10,
                                'transition': 'width 0.3s'}),
            ], style={'width': '100%', 'background': '#ecf0f1',
                      'borderRadius': 10, 'overflow': 'hidden',
                      'marginTop': 10}),
        ])
    
    # 运行中
    gen = opt_progress.get('generation', 0)
    total = opt_progress.get('total', 1)
    best = opt_progress.get('best_fitness', 0)
    obj_type = opt_progress.get('objective', 'eta')
    pct = min(gen / max(total, 1) * 100, 100)
    
    best_str = (f'{best*100:.3f}%' if obj_type == 'eta'
                else f'{best:.3f} kJ/kg')
    
    return html.Div([
        html.Div([
            dcc.Loading(
                id='opt-loading-icon',
                type='circle',
                color='#2980b9',
                children=[
                    html.Span('🔄 优化进行中...',
                             style={'color': '#2980b9', 'fontSize': 15,
                                    'fontWeight': 'bold'}),
                ]
            ),
        ], style={'marginBottom': 6}),
        html.Div([
            html.Span(f'第 {gen}/{total} 代  ({pct:.0f}%)',
                     style={'color': '#2c3e50', 'fontSize': 13,
                            'fontWeight': 'bold'}),
            html.Span(f'  |  当前最佳: {best_str}',
                     style={'color': '#e67e22', 'fontSize': 13,
                            'fontWeight': 'bold', 'marginLeft': 12}),
        ]),
        # 进度条
        html.Div([
            html.Div(style={'width': f'{pct:.0f}%',
                            'height': '20px',
                            'background': 'linear-gradient(90deg, #3498db, #2980b9)',
                            'borderRadius': 10,
                            'transition': 'width 0.3s'}),
        ], style={'width': '100%', 'background': '#ecf0f1',
                  'borderRadius': 10, 'overflow': 'hidden',
                  'marginTop': 10}),
    ])


def _render_opt_results(opt_result):
    """渲染优化结果 (最优参数表 + 收敛曲线 + 完整循环结果)"""
    cfg_key = opt_result.get('cycle_type', 'rankine_basic')
    cfg = CYCLE_CONFIGS.get(cfg_key, {})
    best_params = opt_result.get('best_params', {})
    obj_type = opt_result.get('objective_type', 'eta')
    best_obj = opt_result.get('best_objective', 0)
    initial_params = opt_result.get('initial_params', {})
    
    obj_label = '热效率 η' if obj_type == 'eta' else '净输出功 W_net'
    obj_unit = '%' if obj_type == 'eta' else ' kJ/kg'
    obj_val_str = (f'{best_obj*100:.3f}%' if obj_type == 'eta'
                   else f'{best_obj:.3f}{obj_unit}')
    
    # === 1. 最佳目标值大卡片 ===
    result_header = html.Div([
        html.Div([
            html.Div(f'最优{obj_label}',
                    style={'fontSize': 13, 'color': '#7f8c8d',
                           'marginBottom': 4}),
            html.Div(obj_val_str,
                    style={'fontSize': 32, 'fontWeight': 'bold',
                           'color': '#e67e22'}),
            html.Div(f'({cfg.get("name", cfg_key)})',
                    style={'fontSize': 12, 'color': '#95a5a6',
                           'marginTop': 2}),
        ], style={'flex': 1, 'padding': 16, 'background': '#fdf2e9',
                  'borderRadius': 8, 'borderLeft': '5px solid #e67e22'}),
    ], style={'display': 'flex', 'gap': 16, 'padding': 16,
              'background': '#fef9e7', 'borderRadius': 8,
              'marginBottom': 16, 'alignItems': 'center'})
    
    # === 2. 最优参数表 ===
    param_rows = []
    for p_cfg in cfg.get('params', []):
        key = p_cfg['key']
        label = p_cfg['label']
        unit = p_cfg.get('unit', '')
        opt_val = best_params.get(key, p_cfg.get('default'))
        init_val = initial_params.get(key, p_cfg.get('default'))
        
        is_optimized = key in opt_result.get('optimized_keys', [])
        
        # 变化百分比
        if init_val and init_val != 0:
            pct = (opt_val - init_val) / abs(init_val) * 100
            pct_str = f'{pct:+.2f}%'
            pct_color = '#27ae60' if pct > 0 else (
                '#e74c3c' if pct < 0 else '#7f8c8d')
        else:
            pct_str = '-'
            pct_color = '#7f8c8d'
        
        opt_val_display = f'{opt_val:.4g}'
        init_val_display = f'{init_val:.4g}'
        
        row_style = {}
        if is_optimized:
            row_style['backgroundColor'] = '#fef9e7'
        
        param_rows.append(html.Tr([
            html.Td(label,
                   style={'padding': '8px 12px', 'fontWeight': 'bold',
                          'backgroundColor': '#f8f9fa', **row_style}),
            html.Td(unit, style={'padding': '8px 12px',
                                 'color': '#7f8c8d', **row_style}),
            html.Td(init_val_display,
                   style={'padding': '8px 12px', 'textAlign': 'right',
                          'backgroundColor': '#eaf2f8', **row_style}),
            html.Td(html.Strong(opt_val_display,
                                style={'color': '#935116'}),
                   style={'padding': '8px 12px', 'textAlign': 'right',
                          'backgroundColor': '#fdebd0',
                          'fontWeight': 'bold', **row_style}),
            html.Td(pct_str,
                   style={'padding': '8px 12px', 'textAlign': 'right',
                          'color': pct_color, 'fontWeight': 'bold',
                          **row_style}),
            html.Td(
                html.Span('✓' if is_optimized else '—',
                         style={'color': '#27ae60' if is_optimized else '#bdc3c7',
                                'fontWeight': 'bold'}),
                style={'padding': '8px 12px', 'textAlign': 'center', **row_style}
            ),
        ]))
    
    param_table = html.Div([
        html.H4('📋 最优参数表', style={'marginTop': 0, 'color': '#935116'}),
        html.Div([
            html.Span('📌 高亮行 = 参与优化的参数  |  ',
                     style={'fontSize': 11, 'color': '#7f8c8d'}),
            html.Span('绿↑ = 相比初始值增加  |  红↓ = 相比初始值减少',
                     style={'fontSize': 11, 'color': '#7f8c8d'}),
        ], style={'marginBottom': 8}),
        html.Table([
            html.Thead(html.Tr([
                html.Th('参数名',
                        style={'background': '#e67e22', 'color': 'white',
                               'padding': '10px 12px', 'textAlign': 'left'}),
                html.Th('单位',
                        style={'background': '#e67e22', 'color': 'white',
                               'padding': '10px 12px'}),
                html.Th('初始值',
                        style={'background': '#e67e22', 'color': 'white',
                               'padding': '10px 12px'}),
                html.Th('最优值',
                        style={'background': '#e67e22', 'color': 'white',
                               'padding': '10px 12px'}),
                html.Th('变化率',
                        style={'background': '#e67e22', 'color': 'white',
                               'padding': '10px 12px'}),
                html.Th('优化?',
                        style={'background': '#e67e22', 'color': 'white',
                               'padding': '10px 12px'}),
            ])),
            html.Tbody(param_rows),
        ], style={'width': '100%', 'borderCollapse': 'collapse',
                  'background': 'white', 'fontSize': 13,
                  'marginBottom': 10}),
    ], style={'padding': 16, 'background': '#fdfefe',
              'border': '1px solid #f6b26b',
              'borderRadius': 8, 'marginBottom': 16})
    
    # === 3. 收敛曲线 ===
    gen_best = opt_result.get('generation_best', [])
    gen_avg = opt_result.get('generation_avg', [])
    n_gen = opt_result.get('n_generations', len(gen_best) - 1)
    
    # 准备图表数据
    x_axis = list(range(len(gen_best)))
    
    # 目标值显示格式
    if obj_type == 'eta':
        y_best = [v * 100 for v in gen_best]
        y_avg = [v * 100 for v in gen_avg]
        y_title = f'{obj_label} (%)'
    else:
        y_best = gen_best
        y_avg = gen_avg
        y_title = f'{obj_label} (kJ/kg)'
    
    conv_fig = go.Figure()
    conv_fig.add_trace(go.Scatter(
        x=x_axis, y=y_best, mode='lines+markers',
        name='每代最优', line=dict(color='#e67e22', width=2),
        marker=dict(size=5)
    ))
    conv_fig.add_trace(go.Scatter(
        x=x_axis, y=y_avg, mode='lines',
        name='每代平均', line=dict(color='#3498db', width=2, dash='dot')
    ))
    conv_fig.update_layout(
        title=dict(text=f'📈 收敛曲线 - {cfg.get("name", cfg_key)}',
                   font=dict(size=15, color='#935116')),
        xaxis_title='进化代数',
        yaxis_title=y_title,
        height=420,
        legend=dict(orientation='h', y=1.02, x=0),
        hovermode='x unified',
        plot_bgcolor='#f8f9fa',
    )
    
    # 可行性曲线
    valid_counts = opt_result.get('valid_count_per_gen', [])
    pop_size = opt_result.get('pop_size', 1)
    feas_fig = go.Figure()
    feas_fig.add_trace(go.Bar(
        x=list(range(len(valid_counts))),
        y=[v / pop_size * 100 for v in valid_counts],
        marker_color='#27ae60',
        name='可行解比例',
        opacity=0.75
    ))
    feas_fig.update_layout(
        title=dict(text='✅ 可行解比例 (每代)', font=dict(size=14, color='#1e8449')),
        xaxis_title='进化代数',
        yaxis_title='可行解比例 (%)',
        height=280,
        plot_bgcolor='#f8f9fa',
        yaxis_range=[0, 105],
    )
    
    convergence_section = html.Div([
        html.H4('📈 收敛过程', style={'marginTop': 0, 'color': '#935116'}),
        dcc.Graph(id='opt-conv-figure', figure=conv_fig,
                 style={'marginBottom': 8}),
        dcc.Graph(id='opt-feasibility-figure', figure=feas_fig),
    ], style={'padding': 16, 'background': '#fdfefe',
              'border': '1px solid #f6b26b',
              'borderRadius': 8, 'marginBottom': 16})
    
    # === 4. 完整循环结果 (效率卡片 + T-s图 + 状态点) ===
    # 用最优参数重新计算完整循环
    try:
        best_cycle, best_res = build_cycle(cfg_key, best_params)
        # 效率卡片
        eta = best_res.get('eta', best_res.get('eta_total', 0))
        eta_carnot = best_res.get('eta_carnot', 0)
        w_net = best_res.get('w_net', best_res.get('W_dot_total_kW', 0))
        q_in = best_res.get('q_in', 0)
        w_net_unit = 'kJ/kg' if 'w_net' in best_res else 'kW'
        
        def _make_small_card(title, value, color, unit=''):
            return html.Div([
                html.Div(title, style={'fontSize': 11, 'color': '#7f8c8d',
                                       'marginBottom': 2}),
                html.Div(f'{value}{unit}',
                        style={'fontSize': 18, 'fontWeight': 'bold',
                               'color': color}),
            ], style={'flex': 1, 'background': '#f8f9fa',
                      'padding': 10, 'borderRadius': 6,
                      'borderLeft': f'3px solid {color}'})
        
        opt_cards = [
            _make_small_card('热效率 η', f'{eta*100:.2f}', '#e74c3c', '%'),
            _make_small_card('Carnot效率', f'{eta_carnot*100:.2f}', '#2980b9', '%'),
            _make_small_card('净输出功', f'{w_net:.2f}', '#27ae60', f' {w_net_unit}'),
            _make_small_card('吸热量', f'{q_in:.2f}', '#f39c12', ' kJ/kg' if 'w_net' in best_res else ' kW'),
        ]
        if 'eta_gas' in best_res:
            opt_cards.append(_make_small_card('燃气效率', f"{best_res['eta_gas']*100:.2f}", '#8e44ad', '%'))
            opt_cards.append(_make_small_card('蒸汽效率', f"{best_res['eta_steam']*100:.2f}", '#16a085', '%'))
        
        # T-s图
        ts_fig = plot_Ts_diagram(best_cycle)
        
        # 状态点表
        state_rows = []
        for label, sp in sorted(best_cycle.states.items()):
            state_rows.append({
                '状态点': label,
                '温度 (°C)': round(sp.T - 273.15, 2) if sp.T else None,
                '压力 (MPa)': round(sp.P, 4) if sp.P else None,
                '比焓 (kJ/kg)': round(sp.h, 2) if sp.h else None,
                '比熵 (kJ/kg·K)': round(sp.s, 4) if sp.s else None,
                '干度 x': round(sp.x, 4) if sp.x is not None else '-',
            })
        state_df = pd.DataFrame(state_rows)
        
        cycle_section = html.Div([
            html.H4('♻️ 最优参数下的完整循环结果',
                    style={'marginTop': 0, 'color': '#935116'}),
            # 效率卡片
            html.Div(opt_cards, style={'display': 'flex', 'gap': 8,
                                       'marginBottom': 14, 'flexWrap': 'wrap'}),
            # T-s图
            html.Div([
                html.H5('T-s 图 (温度-比熵)',
                       style={'color': '#2c3e50', 'marginTop': 0,
                              'marginBottom': 6}),
                dcc.Graph(id='opt-best-ts', figure=ts_fig,
                         style={'height': '420px'}),
            ], style={'marginBottom': 14}),
            # 状态点表
            html.Div([
                html.H5('状态点参数',
                       style={'color': '#2c3e50', 'marginTop': 0,
                              'marginBottom': 6}),
                dash_table.DataTable(
                    data=state_df.to_dict('records'),
                    columns=[{'name': c, 'id': c} for c in state_df.columns],
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#16a085',
                                  'color': 'white', 'fontWeight': 'bold'},
                    style_cell={'padding': '5px 10px', 'fontSize': 12,
                               'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'column_id': '状态点'},
                         'fontWeight': 'bold',
                         'backgroundColor': '#f8f9fa'}
                    ]
                ),
            ]),
        ], style={'padding': 16, 'background': '#fdfefe',
                  'border': '1px solid #f6b26b',
                  'borderRadius': 8})
    
    except Exception as e:
        cycle_section = html.Div([
            html.H4('♻️ 最优参数下的完整循环结果',
                    style={'marginTop': 0, 'color': '#935116'}),
            html.Div(f'❌ 重建循环失败: {e}',
                    style={'padding': 20, 'color': '#c0392b',
                           'background': '#fdecea', 'borderRadius': 6}),
        ], style={'padding': 16, 'background': '#fdfefe',
                  'border': '1px solid #f6b26b',
                  'borderRadius': 8})
    
    return html.Div([
        result_header,
        param_table,
        convergence_section,
        cycle_section,
    ], id='opt-results-wrapper')


# ============================================================
# 优化求解器 - 辅助函数: 渲染参数配置面板
# ============================================================

@app.callback(
    Output('opt-param-configs', 'children'),
    Input('opt-cycle-type', 'value')
)
def _render_opt_param_configs(opt_cycle_type):
    """根据循环类型动态渲染参数上下界配置面板"""
    cfg = CYCLE_CONFIGS.get(opt_cycle_type)
    if not cfg:
        return html.Div('请选择循环类型')
    
    children = []
    for p in cfg['params']:
        key = p['key']
        label = p['label']
        unit = p.get('unit', '')
        pmin = p['min']
        pmax = p['max']
        pdefault = p['default']
        pstep = p.get('step', 0.1)
        
        row = html.Div([
            # Checkbox + 参数名
            html.Div([
                dcc.Checklist(
                    id={'type': 'opt-param-enable', 'index': key},
                    options=[{'label': f'  {label}', 'value': '1'}],
                    value=[],
                    style={'fontSize': 12, 'display': 'inline-block'}
                ),
            ], style={'width': '150px', 'flexShrink': 0,
                      'display': 'flex', 'alignItems': 'center'}),
            # 单位
            html.Div(unit,
                     style={'width': '40px', 'flexShrink': 0,
                            'color': '#7f8c8d', 'fontSize': 11,
                            'display': 'flex', 'alignItems': 'center'}),
            # 下界
            html.Div([
                html.Label('下界:', style={'fontSize': 10, 'color': '#7f8c8d',
                                            'marginRight': 3}),
                dcc.Input(
                    id={'type': 'opt-param-min', 'index': key},
                    type='number', min=pmin, max=pmax, step=pstep,
                    value=pmin,
                    style={'width': '70px', 'padding': 3, 'fontSize': 11}
                ),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'flexShrink': 0}),
            # 上界
            html.Div([
                html.Label('上界:', style={'fontSize': 10, 'color': '#7f8c8d',
                                            'marginRight': 3, 'marginLeft': 8}),
                dcc.Input(
                    id={'type': 'opt-param-max', 'index': key},
                    type='number', min=pmin, max=pmax, step=pstep,
                    value=pmax,
                    style={'width': '70px', 'padding': 3, 'fontSize': 11}
                ),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'flexShrink': 0}),
            # 默认值提示
            html.Div(f'(默认 {pdefault:g})',
                     style={'fontSize': 10, 'color': '#aab7b8',
                            'marginLeft': 8, 'display': 'flex',
                            'alignItems': 'center'}),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'padding': '7px 4px',
                  'borderBottom': '1px dashed #ecf0f1',
                  'gap': 2})
        children.append(row)
    
    return html.Div(children)


# ============================================================
# 优化求解器 - 执行优化 (使用后台线程)
# ============================================================

# 线程共享数据 (简单实现, 不支持多个并行会话)
_opt_thread = None
_opt_shared = {
    'progress': None,
    'result': None,
}


def _run_optimization_worker(cfg_key, objective,
                             opt_param_list, fixed_params,
                             pop_size, n_gen, cx_rate, mut_rate):
    """在后台线程中运行优化"""
    def progress_cb(gen, total, best):
        _opt_shared['progress'] = {
            'generation': gen,
            'total': total,
            'best_fitness': best,
            'objective': objective,
        }
    
    try:
        optimizer = GeneticOptimizer(
            cfg_key=cfg_key,
            CYCLE_CONFIGS=CYCLE_CONFIGS,
            objective=objective,
            opt_param_configs=opt_param_list,
            fixed_params=fixed_params,
            pop_size=pop_size,
            n_generations=n_gen,
            crossover_rate=cx_rate,
            mutation_rate=mut_rate,
            progress_callback=progress_cb,
        )
        result = optimizer.run()
        
        # 附加信息
        result['cycle_type'] = cfg_key
        result['optimized_keys'] = [p['key'] for p in opt_param_list]
        result['initial_params'] = fixed_params.copy()
        result['initial_params'].update({
            p['key']: CYCLE_CONFIGS[cfg_key]['params'][i]['default']
            for i, p in enumerate(CYCLE_CONFIGS[cfg_key]['params'])
            if p['key'] not in fixed_params
        })
        # 保存初始参数(优化前) = 当前固定值+默认值
        result['initial_params'] = {
            p['key']: (fixed_params[p['key']]
                       if p['key'] in fixed_params
                       else next((pp['default']
                                  for pp in CYCLE_CONFIGS[cfg_key]['params']
                                  if pp['key'] == p['key']),
                                 p['key']))
            for p in CYCLE_CONFIGS[cfg_key]['params']
        }
        
        _opt_shared['result'] = result
        _opt_shared['progress'] = {
            'generation': n_gen,
            'total': n_gen,
            'best_fitness': result.get('best_objective', 0),
            'objective': objective,
        }
    except Exception as e:
        _opt_shared['result'] = {'error': str(e)}
        _opt_shared['progress'] = {
            'generation': 0,
            'total': 0,
            'best_fitness': 0,
            'objective': objective,
            'error': str(e),
        }


import threading

@app.callback(
    [Output('optimize-running-store', 'data', allow_duplicate=True),
     Output('optimize-progress-store', 'data', allow_duplicate=True)],
    Input('btn-run-optimize', 'n_clicks'),
    [State('opt-cycle-type', 'value'),
     State('opt-objective', 'value'),
     State('opt-pop-size', 'value'),
     State('opt-n-gen', 'value'),
     State('opt-cx-rate', 'value'),
     State('opt-mut-rate', 'value'),
     State({'type': 'opt-param-enable', 'index': ALL}, 'value'),
     State({'type': 'opt-param-min', 'index': ALL}, 'value'),
     State({'type': 'opt-param-max', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def _start_optimization(n_clicks, cfg_key, objective,
                        pop_size, n_gen, cx_rate, mut_rate,
                        enables_all, mins_all, maxs_all):
    global _opt_thread, _opt_shared
    
    if n_clicks == 0:
        raise PreventUpdate
    
    # 获取pattern-matching参数的index列表 (即参数key)
    ctx = callback_context
    enable_keys = [
        inp['id']['index'] for inp in ctx.states_list
        if isinstance(inp['id'], dict) and inp['id']['type'] == 'opt-param-enable'
    ]
    min_keys = [
        inp['id']['index'] for inp in ctx.states_list
        if isinstance(inp['id'], dict) and inp['id']['type'] == 'opt-param-min'
    ]
    max_keys = [
        inp['id']['index'] for inp in ctx.states_list
        if isinstance(inp['id'], dict) and inp['id']['type'] == 'opt-param-max'
    ]
    
    # 构建快速查找字典
    enable_map = dict(zip(enable_keys, enables_all)) if enable_keys else {}
    min_map = dict(zip(min_keys, mins_all)) if min_keys else {}
    max_map = dict(zip(max_keys, maxs_all)) if max_keys else {}
    
    cfg = CYCLE_CONFIGS[cfg_key]
    
    # 构建优化参数列表和固定参数
    opt_param_list = []
    fixed_params = {}
    
    for i, p in enumerate(cfg['params']):
        key = p['key']
        enabled = (key in enable_map and enable_map[key] and len(enable_map[key]) > 0)
        if enabled:
            pmin = min_map.get(key, p['min'])
            pmax = max_map.get(key, p['max'])
            if pmin is None:
                pmin = p['min']
            if pmax is None:
                pmax = p['max']
            # 确保min < max
            if pmin >= pmax:
                pmin = p['min']
                pmax = p['max']
            opt_param_list.append({
                'key': key,
                'min': pmin,
                'max': pmax,
            })
        else:
            # 未勾选 - 使用默认值
            fixed_params[key] = p['default']
    
    # 验证: 至少有一个优化参数
    if len(opt_param_list) == 0:
        return (False, {
            'generation': 0, 'total': 0, 'best_fitness': 0,
            'objective': objective,
            'error': '请至少勾选一个需要优化的参数!'
        })
    
    # 参数默认值校验
    pop_size = int(max(10, pop_size or 50))
    n_gen = int(max(5, n_gen or 80))
    cx_rate = float(np.clip(cx_rate or 0.8, 0.0, 1.0))
    mut_rate = float(np.clip(mut_rate or 0.1, 0.0, 1.0))
    
    # 重置共享数据
    _opt_shared['progress'] = {
        'generation': 0,
        'total': n_gen,
        'best_fitness': 0,
        'objective': objective,
    }
    _opt_shared['result'] = None
    
    # 启动后台线程
    if _opt_thread and _opt_thread.is_alive():
        return (True, _opt_shared['progress'])
    
    _opt_thread = threading.Thread(
        target=_run_optimization_worker,
        args=(cfg_key, objective, opt_param_list, fixed_params,
              pop_size, n_gen, cx_rate, mut_rate),
        daemon=True
    )
    _opt_thread.start()
    
    return (True, _opt_shared['progress'])


# ============================================================
# 优化求解器 - 轮询更新进度和结果
# ============================================================

# 用一个隐藏的Interval组件来轮询
app.layout.children.append(
    dcc.Interval(id='opt-poll-interval', interval=800, n_intervals=0)
)


@app.callback(
    [Output('optimize-progress-store', 'data'),
     Output('optimize-result-store', 'data'),
     Output('optimize-running-store', 'data')],
    Input('opt-poll-interval', 'n_intervals'),
    [State('optimize-running-store', 'data'),
     State('optimize-result-store', 'data'),
     State('optimize-progress-store', 'data')],
    prevent_initial_call=False
)
def _poll_optimization(n_intervals, is_running, cur_result, cur_progress):
    """定期检查后台线程的进度和结果"""
    if not is_running:
        raise PreventUpdate
    
    shared_progress = _opt_shared['progress']
    shared_result = _opt_shared['result']
    
    # 如果有结果了，就标记为结束
    if shared_result is not None:
        if 'error' in shared_result:
            return (
                shared_progress or {
                    'generation': 0, 'total': 0, 'best_fitness': 0,
                    'objective': 'eta', 'error': shared_result['error']
                },
                None,
                False
            )
        return (shared_progress, shared_result, False)
    
    # 否则返回最新进度
    if shared_progress is not None and shared_progress != cur_progress:
        return (shared_progress, cur_result, True)
    
    raise PreventUpdate


# ============================================================
# 优化求解器 - 保存为工况
# ============================================================

@app.callback(
    [Output('cases-store', 'data', allow_duplicate=True),
     Output('opt-save-status', 'children'),
     Output('case-name-input', 'value', allow_duplicate=True)],
    Input('btn-save-opt-case', 'n_clicks'),
    [State('optimize-result-store', 'data'),
     State('cases-store', 'data')],
    prevent_initial_call=True
)
def _save_optimized_case(n_clicks, opt_result, saved_cases):
    """将优化得到的最优参数保存为工况"""
    if n_clicks == 0 or not opt_result:
        raise PreventUpdate
    
    saved_cases = saved_cases or []
    
    if len(saved_cases) >= MAX_CASES:
        return (saved_cases,
                html.Span(f'❌ 工况已达上限{MAX_CASES}',
                         style={'color': '#c0392b'}),
                '')
    
    cfg_key = opt_result.get('cycle_type')
    cfg = CYCLE_CONFIGS.get(cfg_key)
    best_params = opt_result.get('best_params', {})
    obj_type = opt_result.get('objective_type', 'eta')
    best_obj = opt_result.get('best_objective', 0)
    
    obj_label = 'η' if obj_type == 'eta' else 'Wnet'
    obj_val_str = (f'{best_obj*100:.2f}%' if obj_type == 'eta'
                   else f'{best_obj:.2f}')
    
    # 重建循环
    try:
        cycle, res = build_cycle(cfg_key, best_params)
    except Exception as e:
        return (saved_cases,
                html.Span(f'❌ 重建失败: {str(e)[:30]}',
                         style={'color': '#c0392b'}),
                '')
    
    # 工况名
    case_name = f"OPT_{cfg.get('name', '')}_{obj_label}={obj_val_str}"
    case_name = case_name[:30]
    
    # 状态点
    states_dict = {}
    for label, sp in cycle.states.items():
        states_dict[label] = sp.to_dict()
    
    case_id = _generate_case_id()
    case_record = {
        'id': case_id,
        'name': case_name,
        'cycle_type': cfg_key,
        'cycle_name': cfg.get('name', cfg_key),
        'params': best_params,
        'results': _make_serializable(res),
        'states': states_dict,
        'fluid_type': _extract_cycle_fluid_type(cfg_key),
        'color': CASE_COLORS[len(saved_cases) % len(CASE_COLORS)],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    saved_cases.append(case_record)
    
    return (
        saved_cases,
        html.Span(f'✅ 已保存 "{case_name}" ({len(saved_cases)}/{MAX_CASES})',
                 style={'color': '#27ae60', 'fontSize': 11}),
        ''
    )


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("  🔥 工程热力学循环分析与热效率计算工具")
    print("  访问地址: http://127.0.0.1:8050")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=8050)
