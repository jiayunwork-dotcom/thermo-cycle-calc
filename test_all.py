#!/usr/bin/env python3
"""
快速验证测试脚本 - 验证各模块是否正常工作
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def test_steam():
    """测试水蒸气属性计算"""
    print("✅ 测试: 水蒸气属性计算")
    from thermo.steam import steam_state, psat_T, tsat_P, saturation_curve
    
    # 饱和压力
    P_sat_100 = psat_T(373.15)
    print(f"   100°C饱和压力: {P_sat_100:.4f} MPa (≈0.1013 MPa)")
    
    # 饱和温度
    T_sat_1 = tsat_P(1.0)
    print(f"   1MPa饱和温度: {T_sat_1-273.15:.2f} °C (≈179.88 °C)")
    
    # 过热蒸汽
    st = steam_state(T=500+273.15, P=10)
    print(f"   10MPa,500°C: h={st['h']:.2f} kJ/kg, s={st['s']:.4f} kJ/(kg·K)")
    
    # 两相区
    st2 = steam_state(P=1.0, x=0.95)
    print(f"   1MPa,x=0.95: T={st2['T']-273.15:.2f}°C, h={st2['h']:.2f} kJ/kg")
    
    # P,h反推
    st3 = steam_state(P=10, h=3375)
    print(f"   P=10MPa,h=3375反推: T={st3['T']-273.15:.2f}°C")
    
    # 饱和线
    sat = saturation_curve()
    print(f"   饱和线数据点: {len(sat['T'])}")
    
    return True


def test_ideal_gas():
    """测试理想气体"""
    print("✅ 测试: 理想气体属性计算")
    from thermo.ideal_gas import IdealGas
    
    air = IdealGas('air')
    s1 = air.state(T=300, P=0.1)
    print(f"   空气 300K,0.1MPa: h={s1['h']:.2f}, s={s1['s']:.4f}, v={s1['v']:.4f}")
    
    # 等熵压缩
    s2 = air.isentropic_process(s1, P2=1.0)
    print(f"   等熵压缩至1MPa: T={s2['T']:.2f}K")
    
    return True


def test_rankine():
    """测试Rankine循环"""
    print("✅ 测试: Rankine循环")
    from cycles.rankine import RankineCycle, ReheatRankineCycle, RegenerativeRankineCycle
    
    # 基础Rankine
    cyc = RankineCycle(P_boiler=10, T_boiler=500+273.15, P_cond=0.01)
    res = cyc.compute()
    print(f"   基础Rankine: η={res['eta']*100:.3f}%, w_net={res['w_net']:.2f} kJ/kg")
    print(f"   警告: {res.get('warnings', [])}")
    
    # 再热Rankine
    cyc2 = ReheatRankineCycle(P_boiler=15, T_boiler=550+273.15, P_cond=0.008,
                               P_reheat=3, T_reheat=550+273.15)
    res2 = cyc2.compute()
    print(f"   再热Rankine: η={res2['eta']*100:.3f}%, w_net={res2['w_net']:.2f} kJ/kg")
    
    # 回热Rankine
    cyc3 = RegenerativeRankineCycle(P_boiler=12, T_boiler=540+273.15,
                                     P_cond=0.008, P_extract=2)
    res3 = cyc3.compute()
    print(f"   回热Rankine: η={res3['eta']*100:.3f}%, 抽汽比例y={res3['extract_fraction']:.4f}")
    
    return True


def test_brayton():
    """测试Brayton循环"""
    print("✅ 测试: Brayton循环")
    from cycles.brayton import BraytonCycle
    
    cyc = BraytonCycle(P1=0.1, T1=25+273.15, rp=10, T3=1100+273.15)
    res = cyc.compute()
    print(f"   基础Brayton: η={res['eta']*100:.3f}%, 背压功比={res['back_work_ratio']:.3f}")
    
    cyc2 = BraytonCycle(rp=8, T3=1000+273.15, regenerator=True, eta_regenerator=0.85)
    res2 = cyc2.compute()
    print(f"   回热Brayton: η={res2['eta']*100:.3f}%")
    
    return True


def test_ic_engine():
    """测试Otto/Diesel循环"""
    print("✅ 测试: 内燃机循环")
    from cycles.ic_engine import OttoCycle, DieselCycle
    
    otto = OttoCycle(r=8, q_in=1800)
    res1 = otto.compute()
    print(f"   Otto循环: η={res1['eta']*100:.3f}%, MEP={res1['mep']:.3f} MPa")
    
    diesel = DieselCycle(r=16, cutoff=2.0)
    res2 = diesel.compute()
    print(f"   Diesel循环: η={res2['eta']*100:.3f}%, MEP={res2['mep']:.3f} MPa")
    
    return True


def test_combined():
    """测试联合循环"""
    print("✅ 测试: 燃气-蒸汽联合循环")
    from cycles.combined import CombinedCycle
    
    cc = CombinedCycle(rp=14, TIT=1250+273.15, P_steam=10, T_steam=520+273.15)
    res = cc.compute()
    print(f"   CCGT: 燃气η={res['eta_gas']*100:.2f}%, 蒸汽η={res['eta_steam']*100:.2f}%")
    print(f"   联合η={res['eta_total']*100:.2f}%, 理论η={res['eta_combined_theory']*100:.2f}%")
    print(f"   蒸汽/燃气质量比: {res['mass_ratio_steam_gas']:.4f}")
    print(f"   迭代次数: {res.get('converged_iterations', '?')}")
    
    return True


def test_plots():
    """测试绘图"""
    print("✅ 测试: 图表绘制")
    from cycles.rankine import RankineCycle
    from plots import plot_Ts_diagram, plot_Pv_diagram, plot_hs_diagram, plot_exergy_bar
    
    cyc = RankineCycle(P_boiler=10, T_boiler=500+273.15, P_cond=0.01)
    cyc.compute()
    
    fig1 = plot_Ts_diagram(cyc)
    fig2 = plot_Pv_diagram(cyc)
    fig3 = plot_hs_diagram(cyc)
    fig4 = plot_exergy_bar(cyc.results)
    
    print(f"   T-s图 traces: {len(fig1.data)}")
    print(f"   P-v图 traces: {len(fig2.data)}")
    print(f"   h-s图 traces: {len(fig3.data)}")
    print(f"   㶲损失图 traces: {len(fig4.data)}")
    
    return True


def main():
    print("="*60)
    print("  工程热力学循环分析工具 - 模块验证测试")
    print("="*60 + "\n")
    
    tests = [
        ('水蒸气属性', test_steam),
        ('理想气体', test_ideal_gas),
        ('Rankine循环', test_rankine),
        ('Brayton循环', test_brayton),
        ('内燃机循环', test_ic_engine),
        ('联合循环', test_combined),
        ('图表绘制', test_plots),
    ]
    
    passed = 0
    failed = []
    
    for name, func in tests:
        try:
            if func():
                passed += 1
                print(f"\n   ✓ {name} 测试通过\n")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"\n   ✗ {name} 测试失败: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("="*60)
    print(f"  测试结果: {passed}/{len(tests)} 通过")
    if failed:
        print("  失败项目:")
        for name, err in failed:
            print(f"    - {name}: {err}")
    else:
        print("  所有测试通过!可以运行 python run.py 启动Web界面。")
    print("="*60)
    
    return len(failed) == 0


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
