import inspect
import random
import string
import tarfile
from pathlib import Path

def get_project_root() -> Path:
    path = inspect.getabsfile(inspect.currentframe())
    return Path(path).parent.parent.parent.parent.parent.parent

def uncompress(output_path, source_file):
    tarfile_object = tarfile.open(source_file)
    tarfile_object.extractall(output_path)
    tarfile_object.close()

def GenerateRandomID(length = 16, suffix = ''):
    x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length))
    x = x + suffix
    return x
