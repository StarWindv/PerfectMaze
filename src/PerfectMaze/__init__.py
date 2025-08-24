import importlib
import importlib.metadata


_package_name = "PerfectMaze"
__version__   = importlib.metadata.version(_package_name)
__author__    = "星灿长风v(StarWindv)"

_submodules ={
    "HilbertMaze"    : "HilbertMaze",
    "is_perfect_maze": "utils"
}

def __getattr__(name):
    # print("Start dynamic import...")
    if name in _submodules.values():
        return importlib.import_module(f".{name}", package=__name__)

    if name in _submodules:
        module = _submodules[name]
        return getattr(importlib.import_module(f".{module}", package=__name__), name)

    raise ImportError(
        f"cannot import name '{name}' from '{__name__}' " +
        f"({getattr(importlib.import_module(_package_name), "__file__")})\n"
    )