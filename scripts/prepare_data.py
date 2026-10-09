"""Build clean data files (Echoes windows + external sources) into data/processed.

    python scripts/prepare_data.py --config configs/base.yaml [--sources caption kaggle synthetic hardneg]
"""

import argparse
import json

from sed.config import load_config
from sed.data.prepare import prepare


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--sources", nargs="*", default=None, help="external sources to (re)build; default all")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.max_games=3")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    m = prepare(cfg, args.sources)
    print(json.dumps({k: v for k, v in m.items() if k != "split_games"}, indent=1))


if __name__ == "__main__":
    main()
