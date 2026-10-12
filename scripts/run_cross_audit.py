"""Cross-audit of six method families (M1-M6) -> results/loess_six_methods.json."""
import os, runpy
_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
import sys; sys.path.insert(0, _src)
runpy.run_path(os.path.join(_src, "six_methods.py"), run_name="__main__")
