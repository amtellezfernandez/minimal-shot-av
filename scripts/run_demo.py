from __future__ import annotations

from importlib import import_module
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_TARGET_MODULE = "minimal_shot_av.cli.commands.run_demo"
_target = import_module(_TARGET_MODULE)
for _name, _value in vars(_target).items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__"}:
        globals()[_name] = _value

if __name__ == "__main__":
    if hasattr(_target, "main"):
        raise SystemExit(_target.main())
    runpy.run_module(_TARGET_MODULE, run_name="__main__")
