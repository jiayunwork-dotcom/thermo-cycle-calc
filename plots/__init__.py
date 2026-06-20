"""
热力学图绘制模块
支持: T-s图, P-v图, h-s(Mollier)图, 㶲损失柱状图, 参数曲线
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from thermo.steam import saturation_curve, steam_state, T_CRITICAL, T_MAX_STEAM
from thermo.state import StatePoint


def _plotly_layout(title, xlabel, ylabel, height=600, width=900):
    return dict(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title=xlabel, gridcolor='lightgray', showgrid=True),
        yaxis=dict(title=ylabel, gridcolor='lightgray', showgrid=True),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=height,
        width=width,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=30, t=60, b=60),
    )


def plot_saturation_curve_Ts(fig):
    """在T-s图上绘制饱和线 (钟形曲线)"""
    sat = saturation_curve()
    # 饱和液体线
    fig.add_trace(go.Scatter(
        x=sat['sf'], y=sat['T'] - 273.15,
        mode='lines', name='饱和液线',
        line=dict(color='royalblue', width=2, dash='solid'),
        hovertemplate='s=%{x:.3f} kJ/(kg·K)<br>T=%{y:.1f} °C<extra></extra>'
    ))
    # 饱和蒸气线
    fig.add_trace(go.Scatter(
        x=sat['sg'], y=sat['T'] - 273.15,
        mode='lines', name='饱和汽线',
        line=dict(color='royalblue', width=2, dash='solid'),
        hovertemplate='s=%{x:.3f} kJ/(kg·K)<br>T=%{y:.1f} °C<extra></extra>'
    ))
    # 临界点标注
    crit_idx = np.argmax(sat['T'])
    fig.add_annotation(
        x=sat['sf'][crit_idx], y=sat['T'][crit_idx] - 273.15,
        text='临界点', showarrow=True, arrowhead=2,
        ax=40, ay=-20, font=dict(size=10)
    )
    return sat


def plot_saturation_curve_Pv(fig):
    """在P-v图上绘制饱和线"""
    sat = saturation_curve()
    fig.add_trace(go.Scatter(
        x=sat['vf'], y=sat['P'],
        mode='lines', name='饱和液线',
        line=dict(color='royalblue', width=2),
        hovertemplate='v=%{x:.6f} m³/kg<br>P=%{y:.4f} MPa<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=sat['vg'], y=sat['P'],
        mode='lines', name='饱和汽线',
        line=dict(color='royalblue', width=2),
        hovertemplate='v=%{x:.6f} m³/kg<br>P=%{y:.4f} MPa<extra></extra>'
    ))
    return sat


def plot_saturation_curve_hs(fig):
    """在h-s图(Mollier)上绘制饱和线和等压线等"""
    sat = saturation_curve()
    fig.add_trace(go.Scatter(
        x=sat['sf'], y=sat['hf'],
        mode='lines', name='饱和液线',
        line=dict(color='royalblue', width=2),
        hovertemplate='s=%{x:.3f} kJ/(kg·K)<br>h=%{y:.1f} kJ/kg<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=sat['sg'], y=sat['hg'],
        mode='lines', name='饱和汽线',
        line=dict(color='royalblue', width=2),
        hovertemplate='s=%{x:.3f} kJ/(kg·K)<br>h=%{y:.1f} kJ/kg<extra></extra>'
    ))
    # 绘制几条参考等压线
    P_levels = [0.1, 1, 5, 10, 20]
    for P in P_levels:
        s_arr, h_arr = [], []
        for T in np.linspace(tsat_P_safe(P) + 10, T_MAX_STEAM, 30):
            try:
                st = steam_state(T=T, P=P)
                s_arr.append(st['s'])
                h_arr.append(st['h'])
            except:
                continue
        if s_arr:
            fig.add_trace(go.Scatter(
                x=s_arr, y=h_arr,
                mode='lines', name=f'P={P} MPa 等压线',
                line=dict(color='gray', width=1, dash='dot'),
                opacity=0.5, showlegend=(P == P_levels[0])
            ))
    return sat


def tsat_P_safe(P):
    """安全调用饱和温度"""
    from thermo.steam import tsat_P
    try:
        return tsat_P(P)
    except:
        return 400


def plot_Ts_diagram(cycle, title=None):
    """绘制T-s图 (温度-比熵)"""
    fig = go.Figure()
    
    # 绘制饱和线
    plot_saturation_curve_Ts(fig)
    
    # 循环过程路径
    processes = cycle.processes
    states = cycle.states
    
    # 收集路径点
    s_vals, T_vals, labels = [], [], []
    for (l1, l2, ptype) in processes:
        if l1 in states and l2 in states:
            s1, s2 = states[l1].s, states[l2].s
            T1, T2 = states[l1].T - 273.15, states[l2].T - 273.15
            
            # 添加带箭头的线段
            fig.add_trace(go.Scatter(
                x=[s1, s2], y=[T1, T2],
                mode='lines+markers',
                name=f'{l1}→{l2} ({ptype})',
                line=dict(width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate=f'{l1}→{l2}: {ptype}<br>' +
                              's=%{x:.3f}<br>T=%{y:.1f}°C<extra></extra>'
            ))
    
    # 标注状态点
    for label, sp in states.items():
        if sp.s is not None and sp.T is not None:
            s_val, T_val = sp.s, sp.T - 273.15
            annot_text = f"{label}"
            if sp.x is not None:
                annot_text += f"<br>x={sp.x:.3f}"
            fig.add_annotation(
                x=s_val, y=T_val,
                text=annot_text,
                showarrow=True, arrowhead=1, arrowsize=1,
                ax=15, ay=-15,
                font=dict(size=11, color='darkred'),
                bgcolor='rgba(255,255,255,0.7)'
            )
    
    # 布局
    if not title:
        title = f'{cycle.name} - T-s 图 (温度-比熵)'
    
    fig.update_layout(
        **_plotly_layout(title, 
                        xlabel='比熵 s [kJ/(kg·K)]', 
                        ylabel='温度 T [°C]')
    )
    return fig


def plot_Pv_diagram(cycle, title=None):
    """绘制P-v图 (压力-比体积)"""
    fig = go.Figure()
    
    plot_saturation_curve_Pv(fig)
    fig.update_yaxes(type='log')
    fig.update_xaxes(type='log')
    
    states = cycle.states
    processes = cycle.processes
    
    for (l1, l2, ptype) in processes:
        if l1 in states and l2 in states:
            s1, s2 = states[l1], states[l2]
            if s1.v and s2.v and s1.P and s2.P:
                fig.add_trace(go.Scatter(
                    x=[s1.v, s2.v], y=[s1.P, s2.P],
                    mode='lines+markers',
                    name=f'{l1}→{l2} ({ptype})',
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate=f'{l1}→{l2}: {ptype}<br>' +
                                  'v=%{x:.6f} m³/kg<br>P=%{y:.4f} MPa<extra></extra>'
                ))
    
    for label, sp in states.items():
        if sp.v and sp.P:
            fig.add_annotation(
                x=sp.v, y=sp.P,
                text=label,
                showarrow=True, arrowhead=1,
                ax=15, ay=-15,
                font=dict(size=11, color='darkred'),
                bgcolor='rgba(255,255,255,0.7)'
            )
    
    if not title:
        title = f'{cycle.name} - P-v 图 (压力-比体积, 对数坐标)'
    
    fig.update_layout(
        **_plotly_layout(title,
                        xlabel='比体积 v [m³/kg] (对数)',
                        ylabel='压力 P [MPa] (对数)')
    )
    return fig


def plot_hs_diagram(cycle, title=None):
    """绘制h-s图 (Mollier图, 比焓-比熵)"""
    fig = go.Figure()
    
    plot_saturation_curve_hs(fig)
    
    states = cycle.states
    processes = cycle.processes
    
    for (l1, l2, ptype) in processes:
        if l1 in states and l2 in states:
            s1, s2 = states[l1], states[l2]
            if s1.h and s2.h and s1.s and s2.s:
                fig.add_trace(go.Scatter(
                    x=[s1.s, s2.s], y=[s1.h, s2.h],
                    mode='lines+markers',
                    name=f'{l1}→{l2} ({ptype})',
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate=f'{l1}→{l2}: {ptype}<br>' +
                                  's=%{x:.3f}<br>h=%{y:.1f} kJ/kg<extra></extra>'
                ))
    
    for label, sp in states.items():
        if sp.h and sp.s:
            fig.add_annotation(
                x=sp.s, y=sp.h,
                text=label,
                showarrow=True, arrowhead=1,
                ax=15, ay=-15,
                font=dict(size=11, color='darkred'),
                bgcolor='rgba(255,255,255,0.7)'
            )
    
    if not title:
        title = f'{cycle.name} - h-s 图 (Mollier图, 比焓-比熵)'
    
    fig.update_layout(
        **_plotly_layout(title,
                        xlabel='比熵 s [kJ/(kg·K)]',
                        ylabel='比焓 h [kJ/kg]')
    )
    return fig


def plot_exergy_bar(results_dict, title='㶲损失分布'):
    """绘制各组件㶲损失柱状图"""
    ex_d = results_dict.get('exergy_destruction', {})
    if not ex_d:
        fig = go.Figure()
        fig.update_layout(title='无㶲损失数据')
        return fig
    
    components = list(ex_d.keys())
    values = list(ex_d.values())
    
    # 按损失排序
    sorted_idx = np.argsort(values)[::-1]
    components = [components[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]
    
    # 颜色映射
    colors = ['#e74c3c' if i == 0 else '#3498db' if v > max(values)*0.3 
              else '#95a5a6' for i, v in enumerate(values)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=components, y=values,
        marker_color=colors,
        text=[f'{v:.2f} kJ/kg' for v in values],
        textposition='auto',
        hovertemplate='%{x}<br>㶲损失=%{y:.2f} kJ/kg<extra></extra>'
    ))
    
    total = sum(values)
    fig.add_annotation(
        x=0.5, y=0.95, xref='paper', yref='paper',
        text=f'总㶲损失: {total:.2f} kJ/kg',
        showarrow=False,
        font=dict(size=13, color='darkred'),
        bgcolor='rgba(255,230,230,0.8)'
    )
    
    # 标注最大损失
    if components:
        fig.add_annotation(
            x=components[0], y=values[0],
            text='← 最薄弱环节',
            showarrow=True, arrowhead=2,
            ax=80, ay=0,
            font=dict(color='darkred', size=11)
        )
    
    fig.update_layout(
        **_plotly_layout(title, 
                        xlabel='组件',
                        ylabel='㶲损失 [kJ/kg]')
    )
    return fig


def plot_parametric_curve(param_values, eta_values, wnet_values,
                          param_name='参数', param_unit='',
                          title='参数化分析曲线'):
    """绘制参数-效率/净功曲线"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=param_values, y=eta_values * 100, 
                   mode='lines+markers', name='热效率',
                   line=dict(color='#e74c3c', width=3),
                   marker=dict(size=8),
                   hovertemplate=f'{param_name}=%{{x}}{param_unit}<br>' +
                                 '效率=%{y:.2f}%<extra></extra>'),
        secondary_y=False,
    )
    
    if wnet_values is not None:
        fig.add_trace(
            go.Scatter(x=param_values, y=wnet_values,
                       mode='lines+markers', name='净功',
                       line=dict(color='#3498db', width=3),
                       marker=dict(size=8),
                       hovertemplate=f'{param_name}=%{{x}}{param_unit}<br>' +
                                     '净功=%{y:.2f} kJ/kg<extra></extra>'),
            secondary_y=True,
        )
    
    fig.update_xaxes(title_text=f'{param_name} [{param_unit}]')
    fig.update_yaxes(title_text='热效率 [%]', secondary_y=False, color='#e74c3c')
    fig.update_yaxes(title_text='净功 [kJ/kg]', secondary_y=True, color='#3498db')
    
    fig.update_layout(
        **_plotly_layout(title, 
                        xlabel=f'{param_name} [{param_unit}]',
                        ylabel='')
    )
    return fig


def plot_2d_contour(x_vals, y_vals, z_mtx, 
                    x_name='X', y_name='Y', z_name='效率',
                    x_unit='', y_unit='', z_unit='%',
                    title='二维等值线图'):
    """绘制二维等值线图 (多参数扫描)"""
    fig = go.Figure(data=go.Contour(
        z=z_mtx,
        x=x_vals,
        y=y_vals,
        colorscale='Viridis',
        contours=dict(
            coloring='heatmap',
            showlabels=True,
            labelfont=dict(size=12, color='white'),
        ),
        colorbar=dict(title=f'{z_name} [{z_unit}]'),
        hovertemplate=f'{x_name}=%{{x}}{x_unit}<br>' +
                      f'{y_name}=%{{y}}{y_unit}<br>' +
                      f'{z_name}=%{{z:.2f}} {z_unit}<extra></extra>'
    ))
    
    fig.update_layout(
        **_plotly_layout(title,
                        xlabel=f'{x_name} [{x_unit}]',
                        ylabel=f'{y_name} [{y_unit}]')
    )
    return fig


def plot_superimposed_Ts(cycles_data, title='多工况T-s图对比'):
    """
    在同一张T-s图上叠加多个工况的循环路径
    cycles_data: list of dict
      {
        'name': 工况名,
        'cycle': cycle对象 (有states和processes属性),
        'fluid_type': 'water' 或 'gas'  (决定线型)
        'color': 颜色
      }
    """
    fig = go.Figure()
    
    # 绘制饱和线 (只绘制一次,用于水蒸汽工况)
    has_water = any(cd['fluid_type'] == 'water' for cd in cycles_data)
    if has_water:
        plot_saturation_curve_Ts(fig)
    
    # 预设颜色
    default_colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', 
                      '#8e44ad', '#16a085', '#2c3e50', '#d35400']
    
    for i, cd in enumerate(cycles_data):
        name = cd.get('name', f'工况{i+1}')
        cycle = cd['cycle']
        fluid_type = cd.get('fluid_type', 'water')
        color = cd.get('color', default_colors[i % len(default_colors)])
        
        # 线型: 水用实线, 气体用虚线
        dash_style = 'solid' if fluid_type == 'water' else 'dash'
        line_width = 3 if fluid_type == 'water' else 2.5
        
        states = cycle.states
        processes = cycle.processes
        
        # 绘制过程路径
        for (l1, l2, ptype) in processes:
            if l1 in states and l2 in states:
                s1, s2 = states[l1].s, states[l2].s
                T1, T2 = states[l1].T - 273.15, states[l2].T - 273.15
                
                show_legend = (ptype == processes[0][2])  # 只在第一个过程显示图例
                
                fig.add_trace(go.Scatter(
                    x=[s1, s2], y=[T1, T2],
                    mode='lines+markers',
                    name=name,
                    legendgroup=name,
                    showlegend=show_legend,
                    line=dict(color=color, width=line_width, dash=dash_style),
                    marker=dict(size=7, color=color, symbol='circle'),
                    hovertemplate=f'<b>{name}</b><br>{l1}→{l2}: {ptype}<br>' +
                                  's=%{x:.3f} kJ/(kg·K)<br>T=%{y:.1f}°C<extra></extra>'
                ))
        
        # 标注状态点 (只标注第一个和最后一个,避免拥挤)
        key_labels = sorted(states.keys())
        for label in [key_labels[0], key_labels[-1]] if len(key_labels) >= 2 else key_labels:
            sp = states[label]
            if sp.s is not None and sp.T is not None:
                fig.add_annotation(
                    x=sp.s, y=sp.T - 273.15,
                    text=f"<b>{name}</b><br>{label}",
                    showarrow=True, arrowhead=1, arrowsize=1,
                    ax=15, ay=-15,
                    font=dict(size=10, color=color),
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor=color, borderwidth=1,
                )
    
    # 图例说明线型含义
    if has_water and any(cd['fluid_type'] == 'gas' for cd in cycles_data):
        fig.add_annotation(
            x=0.02, y=0.02, xref='paper', yref='paper',
            text='— 实线: 水/水蒸气工质 &nbsp;&nbsp; -- 虚线: 气体工质',
            showarrow=False,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#ccc', borderwidth=1,
        )
    
    fig.update_layout(
        **_plotly_layout(title,
                        xlabel='比熵 s [kJ/(kg·K)]',
                        ylabel='温度 T [°C]')
    )
    fig.update_layout(
        legend=dict(x=0.98, y=0.98, xanchor='right', yanchor='top',
                   bgcolor='rgba(255,255,255,0.9)',
                   bordercolor='#ccc', borderwidth=1)
    )
    return fig


def plot_comparison_radar(cases_data, title='多工况综合性能雷达图'):
    """
    雷达图: 五维度对比
      - 热效率 η
      - 净功 w_net (归一化)
      - 㶲效率
      - 1 - 放热率 (q_out/q_in → 越低越好,取倒数)
      - 紧凑性 (1/比体积变化)
    
    cases_data: list of dict
      {
        'name': 工况名,
        'eta': 热效率,
        'w_net': 净功 kJ/kg,
        'exergy_eff': 㶲效率,
        'heat_rejection_ratio': q_out/q_in,
        'compactness': 紧凑性指标 (1/Δv),
        'color': 颜色 (可选)
      }
    """
    default_colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12',
                      '#8e44ad', '#16a085', '#2c3e50', '#d35400']
    
    categories = ['热效率 η', '净功输出', '㶲效率', '热量利用率\n(1-q_out/q_in)', '紧凑性']
    
    fig = go.Figure()
    
    # 归一化: 找出每个维度的最大值作为参考 (紧凑性可能有特殊处理)
    max_vals = {
        'eta': max(max(cd.get('eta', 0) for cd in cases_data), 1e-6),
        'w_net': max(max(abs(cd.get('w_net', 0)) for cd in cases_data), 1e-6),
        'exergy_eff': max(max(cd.get('exergy_eff', 0) for cd in cases_data), 1e-6),
        'heat_util': max(max(max(1 - cd.get('heat_rejection_ratio', 0), 0) for cd in cases_data), 1e-6),
        'compactness': max(max(cd.get('compactness', 0) for cd in cases_data), 1e-6),
    }
    
    for i, cd in enumerate(cases_data):
        name = cd.get('name', f'工况{i+1}')
        color = cd.get('color', default_colors[i % len(default_colors)])
        
        # 各维度值 (归一化到0-1, 再转为百分比显示)
        eta_norm = cd.get('eta', 0) / max_vals['eta'] * 100
        wnet_norm = abs(cd.get('w_net', 0)) / max_vals['w_net'] * 100
        exergy_norm = cd.get('exergy_eff', 0) / max_vals['exergy_eff'] * 100
        heat_util_norm = max(1 - cd.get('heat_rejection_ratio', 0), 0) / max_vals['heat_util'] * 100
        compact_norm = cd.get('compactness', 0) / max_vals['compactness'] * 100
        
        r_values = [eta_norm, wnet_norm, exergy_norm, heat_util_norm, compact_norm]
        
        # 闭合雷达图
        r_values_closed = r_values + [r_values[0]]
        theta_closed = categories + [categories[0]]
        
        # 实际值hover显示
        hover_texts = [
            f'{name}<br>热效率: {cd.get("eta",0)*100:.2f}%',
            f'{name}<br>净功: {cd.get("w_net",0):.2f} kJ/kg',
            f'{name}<br>㶲效率: {cd.get("exergy_eff",0)*100:.2f}%',
            f'{name}<br>热量利用率: {max(1-cd.get("heat_rejection_ratio",0),0)*100:.2f}%',
            f'{name}<br>紧凑性: {cd.get("compactness",0):.4f}',
            f'{name}<br>热效率: {cd.get("eta",0)*100:.2f}%',
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=theta_closed,
            fill='toself',
            name=name,
            fillcolor=color,
            opacity=0.25,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            hovertext=hover_texts,
            hoverinfo='text',
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 105],
                tickfont=dict(size=10),
                gridcolor='lightgray',
                title=dict(text='归一化值 (%)', font=dict(size=10))
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#2c3e50'),
                gridcolor='lightgray',
                linecolor='#2c3e50',
            ),
            bgcolor='#fafafa'
        ),
        title=dict(text=title, font=dict(size=16), x=0.5),
        height=600,
        width=800,
        legend=dict(x=0.98, y=0.02, xanchor='right', yanchor='bottom',
                   bgcolor='rgba(255,255,255,0.9)',
                   bordercolor='#ccc', borderwidth=1),
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=80, b=60),
    )
    return fig
