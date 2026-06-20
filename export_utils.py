"""
导出功能模块
- 状态点参数CSV
- Plotly图表SVG导出
- PDF报告 (使用reportlab或HTML转PDF)
"""

import os
import io
import base64
import pandas as pd
from datetime import datetime


def export_states_csv(cycle, filepath=None):
    """导出状态点参数为CSV"""
    import pandas as pd
    rows = []
    for label, sp in cycle.states.items():
        d = sp.to_dict()
        d['T_°C'] = d['T'] - 273.15 if d.get('T') else None
        d['P_MPa'] = d.get('P')
        d['h_kJ/kg'] = d.get('h')
        d['s_kJ/kgK'] = d.get('s')
        d['v_m3/kg'] = d.get('v')
        d['x'] = d.get('x')
        d['region'] = d.get('region')
        rows.append(d)
    
    df = pd.DataFrame(rows)[['label', 'T_°C', 'P_MPa', 'h_kJ/kg', 's_kJ/kgK', 
                               'v_m3/kg', 'x', 'region', 'fluid']]
    df.columns = ['状态点', '温度(°C)', '压力(MPa)', '比焓(kJ/kg)', 
                  '比熵(kJ/kg·K)', '比体积(m³/kg)', '干度', '区域', '工质']
    
    if filepath:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath
    else:
        return df.to_csv(index=False, encoding='utf-8-sig')


def export_results_csv(cycle, filepath=None):
    """导出计算结果汇总为CSV"""
    res = cycle.results
    rows = []
    for key, val in res.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                rows.append({'项目': f'{key}.{k2}', '数值': v2})
        elif isinstance(val, list):
            rows.append({'项目': key, '数值': '; '.join(map(str, val))})
        else:
            rows.append({'项目': key, '数值': val})
    
    df = pd.DataFrame(rows)
    if filepath:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return filepath
    else:
        return df.to_csv(index=False, encoding='utf-8-sig')


def export_figure_svg(fig, filepath):
    """导出Plotly图表为SVG矢量格式"""
    try:
        fig.write_image(filepath, format='svg', engine='kaleido')
        return True, filepath
    except Exception as e:
        # 备用方法: 用plotly.io
        try:
            import plotly.io as pio
            svg_content = pio.to_image(fig, format='svg')
            with open(filepath, 'wb') as f:
                f.write(svg_content)
            return True, filepath
        except Exception as e2:
            return False, str(e) + '; ' + str(e2)


def export_figure_png(fig, filepath, scale=2):
    """导出Plotly图表为PNG"""
    try:
        fig.write_image(filepath, format='png', scale=scale, engine='kaleido')
        return True, filepath
    except Exception as e:
        return False, str(e)


def figure_to_svg_bytes(fig):
    """将图表转换为SVG字节"""
    import plotly.io as pio
    try:
        return pio.to_image(fig, format='svg')
    except:
        return fig.to_image(format='svg')


def figure_to_base64_png(fig):
    """图表转base64 PNG (用于PDF)"""
    import plotly.io as pio
    try:
        img_bytes = pio.to_image(fig, format='png', scale=2)
        return base64.b64encode(img_bytes).decode()
    except:
        return None


def generate_pdf_report(cycle, figures, filepath, title=None):
    """
    生成PDF报告 (使用reportlab)
    figures: dict {name: plotly_figure}
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm, cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image, PageBreak)
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return False, "reportlab 未安装"
    
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    
    story = []
    styles = getSampleStyleSheet()
    
    # 标题
    title_text = title or f'{cycle.name} - 热力学循环分析报告'
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=30,
        textColor=colors.HexColor('#2c3e50')
    )
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                           styles['Normal']))
    story.append(Spacer(1, 0.5*cm))
    
    # 效率汇总
    story.append(Paragraph('<b>一、效率汇总</b>', styles['Heading2']))
    res = cycle.results
    
    eff_data = [
        ['指标', '数值'],
        ['热效率 η', f"{res.get('eta', res.get('eta_total', 0))*100:.3f}%"],
        ['Carnot效率', f"{res.get('eta_carnot', 0)*100:.3f}%"],
        ['净输出功', f"{res.get('w_net', res.get('W_dot_total_kW', 0)):.3f} " + 
         ('kW' if 'W_dot' in str(res.get('w_net', '')) else 'kJ/kg')],
        ['吸热量', f"{res.get('q_in', 0):.3f} kJ/kg"],
        ['放热量', f"{res.get('q_out', 0):.3f} kJ/kg"],
    ]
    
    # 联合循环
    if 'eta_gas' in res:
        eff_data.extend([
            ['燃气循环效率', f"{res['eta_gas']*100:.3f}%"],
            ['蒸汽循环效率', f"{res['eta_steam']*100:.3f}%"],
            ['理论联合效率', f"{res.get('eta_combined_theory', 0)*100:.3f}%"],
        ])
    
    t = Table(eff_data, colWidths=[8*cm, 6*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.8*cm))
    
    # 警告信息
    warnings = res.get('warnings', [])
    if warnings:
        story.append(Paragraph('<b>警告</b>', styles['Heading2']))
        for w in warnings:
            story.append(Paragraph(f'⚠ {w}', styles['Normal']))
        story.append(Spacer(1, 0.5*cm))
    
    # 状态点参数表
    story.append(Paragraph('<b>二、状态点参数</b>', styles['Heading2']))
    
    state_data = [['状态点', 'T(°C)', 'P(MPa)', 'h(kJ/kg)', 
                   's(kJ/kg·K)', 'v(m³/kg)', '干度x', '区域']]
    
    for label, sp in cycle.states.items():
        row = [
            label,
            f'{sp.T - 273.15:.2f}' if sp.T else '-',
            f'{sp.P:.4f}' if sp.P else '-',
            f'{sp.h:.2f}' if sp.h else '-',
            f'{sp.s:.4f}' if sp.s else '-',
            f'{sp.v:.6f}' if sp.v else '-',
            f'{sp.x:.4f}' if sp.x is not None else '-',
            str(sp.region),
        ]
        state_data.append(row)
    
    t2 = Table(state_data, colWidths=[1.5*cm, 2.2*cm, 2*cm, 2.5*cm, 
                                       2.5*cm, 2.5*cm, 1.5*cm, 1.8*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.gray),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.8*cm))
    
    # 㶲损失分析
    ex_d = res.get('exergy_destruction', {})
    if ex_d:
        story.append(Paragraph('<b>三、㶲损失分析</b>', styles['Heading2']))
        ex_data = [['组件', '㶲损失(kJ/kg)', '占比(%)']]
        total = sum(ex_d.values())
        for comp, val in sorted(ex_d.items(), key=lambda x: -x[1]):
            ex_data.append([
                comp,
                f'{val:.3f}',
                f'{val/total*100:.2f}%' if total > 0 else '0%'
            ])
        ex_data.append(['合计', f'{total:.3f}', '100%'])
        t3 = Table(ex_data, colWidths=[6*cm, 4*cm, 4*cm])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.gray),
            ('BACKGROUND', (0, 1), (-1, -2), colors.whitesmoke),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#fdebd0')),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        story.append(t3)
        story.append(Spacer(1, 0.5*cm))
    
    # 图表
    if figures:
        story.append(PageBreak())
        story.append(Paragraph('<b>四、热力学图</b>', styles['Heading2']))
        
        for fig_name, fig in figures.items():
            story.append(Paragraph(fig_name, styles['Heading3']))
            try:
                # 转PNG嵌入
                png_b64 = figure_to_base64_png(fig)
                if png_b64:
                    img_data = base64.b64decode(png_b64)
                    img = Image(io.BytesIO(img_data), width=16*cm, height=10*cm)
                    story.append(img)
                else:
                    story.append(Paragraph('[图表渲染失败]', styles['Normal']))
            except Exception as e:
                story.append(Paragraph(f'[图表错误: {e}]', styles['Normal']))
            story.append(Spacer(1, 0.5*cm))
    
    try:
        doc.build(story)
        return True, filepath
    except Exception as e:
        return False, str(e)
