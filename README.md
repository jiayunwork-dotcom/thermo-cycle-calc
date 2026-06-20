# 🔥 工程热力学循环分析与热效率计算工具

基于Python + Dash + Plotly开发的交互式热力学循环分析工具，支持多种动力循环的状态点参数计算、效率分析、热力学图可视化和㶲分析。

## 功能特性

### 工质属性计算
- **水蒸气 (IAPWS-IF97简化版)**:
  - 区域1: 压缩液区 (T,P → h,s,ρ)
  - 区域2: 过热蒸气区 (T,P → h,s,v)
  - 区域3: 两相湿蒸气区 (P,x 或 T,x → 混合h,s,v)
  - 支持多种参数反推: P,h / P,s / T,s / P,T / h,s 等

- **理想气体模型**:
  - 支持空气、氮气、氩气、氦气、CO₂、甲烷等
  - 支持多变过程、等熵过程、考虑效率的压缩/膨胀过程

### 循环类型
| 循环类型 | 说明 |
|---------|------|
| **基础Rankine** | 泵→锅炉→汽轮机→冷凝器 |
| **再热Rankine** | 高压缸→再热器→低压缸 |
| **回热Rankine** | 开式一级给水加热器，抽汽比例迭代求解 |
| **基础Brayton** | 压气机→燃烧室→涡轮 |
| **回热Brayton** | 带回热器，排气预热进气 |
| **中间冷却Brayton** | 多级压缩+中间冷却 |
| **Otto循环** | 汽油机等容加热循环 |
| **Diesel循环** | 柴油机等压加热循环 |
| **CCGT联合循环** | 燃气+蒸汽联合循环，HRSG耦合迭代 |

### 热力学图
- **T-s 图** (温度-比熵): 饱和线钟形曲线，面积直观表示热量
- **P-v 图** (压力-比体积): 对数坐标
- **h-s 图** (Mollier图): 带参考等压线
- **㶲损失柱状图**: 识别热力学薄弱环节

### 分析功能
- **效率计算**: 热效率、Carnot效率、背压功比、净功、吸/放热量
- **第二定律分析**: 各组件熵产、㶲损失分布、㶲效率
- **参数化扫描**: 单参数效率/净功曲线
- **二维等值线**: 双参数扫描 (预留接口)
- **智能警告**: 末级湿度过大(<0.88)、温度超上限等

### 导出功能
- 📄 **PDF报告**: 状态点参数表、效率汇总、所有热力学图
- 📊 **SVG矢量图**: 各图表高清晰下载
- 📋 **CSV数据**: 状态点参数、计算结果

## 快速开始

### 环境要求
- Python 3.8+
- 依赖包: dash, plotly, numpy, scipy, pandas, kaleido, reportlab

### 安装
```bash
pip install -r requirements.txt
```

### 启动
```bash
python run.py
# 或
python app.py
```

然后在浏览器访问: http://127.0.0.1:8050

### 运行测试
```bash
python test_all.py
```

## 项目结构

```
thermo-cycle-calc/
├── app.py                  # Dash主程序
├── run.py                  # 启动脚本
├── test_all.py             # 模块测试
├── requirements.txt        # 依赖
├── export_utils.py         # 导出工具
├── thermo/                 # 工质属性包
│   ├── __init__.py
│   ├── steam.py            # 水蒸气IAPWS-IF97简化版
│   ├── ideal_gas.py        # 理想气体模型
│   └── state.py            # 状态点统一接口
├── cycles/                 # 循环计算包
│   ├── __init__.py
│   ├── rankine.py          # Rankine循环(基础/再热/回热)
│   ├── brayton.py          # Brayton循环(基础/回热/中冷)
│   ├── ic_engine.py        # Otto/Diesel内燃机循环
│   └── combined.py         # CCGT联合循环
├── analysis/               # 分析工具
│   └── __init__.py         # 参数化扫描、㶲分析
└── plots/                  # 绘图工具
    └── __init__.py         # T-s/P-v/h-s/㶲损失/参数曲线
```

## 业务约束
| 项目 | 约束值 |
|-----|--------|
| 水蒸气温度上限 | 650°C |
| 空气温度上限 | 1500°C |
| 末级最低干度 | 0.88 |
| 水蒸气焓误差 | ≤0.5 kJ/kg (简化版约≤50kJ/kg，可进一步校准) |
| 迭代收敛判据 | 相对误差 < 0.001 |

## 使用示例

### Python API直接调用
```python
from cycles.rankine import RankineCycle
from plots import plot_Ts_diagram

# 创建循环
cyc = RankineCycle(
    P_boiler=10,       # MPa
    T_boiler=500+273.15,  # K
    P_cond=0.01,       # MPa
    eta_pump=0.85,
    eta_turbine=0.90
)

# 计算
res = cyc.compute()
print(f"热效率: {res['eta']*100:.2f}%")
print(f"净功: {res['w_net']:.2f} kJ/kg")

# 绘制T-s图
fig = plot_Ts_diagram(cyc)
fig.show()

# 查看警告
print("警告:", res.get('warnings', []))
```

## 技术栈
- **前端交互**: Dash (Plotly)
- **数值计算**: NumPy, SciPy
- **数据处理**: Pandas
- **图表渲染**: Plotly Graph Objects
- **PDF生成**: ReportLab
- **图片导出**: Kaleido
