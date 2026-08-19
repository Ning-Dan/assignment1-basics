import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:  # not installed as a package
    __version__ = "0.0.0"

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe
