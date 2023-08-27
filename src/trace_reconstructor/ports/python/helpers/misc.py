import inspect
from pathlib import Path

def get_project_root() -> Path:
    path = inspect.getabsfile(inspect.currentframe())
    return Path(path).parent.parent.parent.parent.parent.parent
