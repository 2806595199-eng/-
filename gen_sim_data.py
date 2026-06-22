import sys
from training import gen_sim_data as _module


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    _module.generate(a.n, a.seed)
else:
    sys.modules[__name__] = _module
