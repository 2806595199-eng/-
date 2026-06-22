import sys
from training import train as _module


if __name__ == "__main__":
    import argparse

    from core import config as cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()
    _module.main(args.data, args.device, args.output_dir)
else:
    sys.modules[__name__] = _module
