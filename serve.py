import sys
from serving import serve as _module


if __name__ == "__main__":
    _module.main()
else:
    sys.modules[__name__] = _module
