import argparse
from dataset import build_cache
ap = argparse.ArgumentParser(); ap.add_argument("--data_dir", required=True); ap.add_argument("--out", default="cache.npz")
a = ap.parse_args(); build_cache(a.data_dir, a.out)
