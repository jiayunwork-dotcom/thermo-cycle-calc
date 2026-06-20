#!/usr/bin/env python3
"""
启动脚本 - 工程热力学循环分析工具
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def check_dependencies():
    """检查依赖"""
    missing = []
    required = ['dash', 'plotly', 'numpy', 'scipy', 'pandas']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("缺少以下依赖包:")
        for m in missing:
            print(f"  - {m}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    return True

def main():
    print("=" * 60)
    print("  🔥 工程热力学循环分析与热效率计算工具")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    from app import app
    
    print("\n启动Web界面...")
    print("访问地址: http://127.0.0.1:8050")
    print("按 Ctrl+C 停止\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        print("\n已停止。")

if __name__ == '__main__':
    main()
