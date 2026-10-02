"""Download all public data used by the project into ``data/raw`` (nothing is committed to git).

Sources (pinned commits for reproducibility):

* SoccerNet-Echoes transcripts  — github.com/SoccerNet/sn-echoes (main)       ~530 MB
* Labels-v2 + Caption mirror    — github.com/SoccerNet/sn-echoes (sushant)    summaries/ + SN-echoes-lang.csv
* Kaggle Football Events        — Kaggle API if credentials exist, else a public GitHub mirror of the
                                  same zip (github.com/DakshS07/xG_Football, Football_events.zip)
* Official Labels-v2 / Labels-caption (optional, ``--official``) — SoccerNet pip package downloader.

Usage:  python -m sed.data.download --root data/raw [--official] [--skip-kaggle]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ECHOES_URL = "https://github.com/SoccerNet/sn-echoes"
ECHOES_MAIN_COMMIT = "7105a85b7a8c1c000a31a30d0c29c388105c3de5"
ECHOES_SUSHANT_COMMIT = "b3565c116ef5a287106e5fe7b47cf626921534c8"
KAGGLE_DATASET = "secareanualin/football-events"
KAGGLE_MIRROR_URL = "https://github.com/DakshS07/xG_Football"
KAGGLE_MIRROR_COMMIT = "c477b072c8d308d01df52926a8f8bdbda7b2481a"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _sparse_clone(url: str, dest: Path, commit: str, paths: list[str], branch: str | None = None) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {dest} exists")
        return
    dest.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-q"], cwd=dest)
    _run(["git", "remote", "add", "origin", url], cwd=dest)
    _run(["git", "config", "core.sparseCheckout", "true"], cwd=dest)
    _run(["git", "sparse-checkout", "init", "--cone"], cwd=dest)
    _run(["git", "sparse-checkout", "set", *paths], cwd=dest)
    try:
        _run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], cwd=dest)
    except subprocess.CalledProcessError:  # server refuses fetching by sha -> fall back to branch tip
        _run(["git", "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", branch or "main"], cwd=dest)
    _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


def download_echoes(root: Path) -> None:
    _sparse_clone(ECHOES_URL, root / "sn-echoes", ECHOES_MAIN_COMMIT, ["Dataset"], branch="main")


def download_mirror(root: Path) -> None:
    _sparse_clone(ECHOES_URL, root / "sn-echoes-sushant", ECHOES_SUSHANT_COMMIT, ["summaries"], branch="sushant")


def download_kaggle(root: Path) -> None:
    dest = root / "football-events"
    if (dest / "events.csv").exists():
        print(f"[skip] {dest} exists")
        return
    dest.mkdir(parents=True, exist_ok=True)
    has_creds = (Path.home() / ".kaggle" / "kaggle.json").exists() or os.environ.get("KAGGLE_USERNAME")
    if has_creds and shutil.which("kaggle"):
        try:
            _run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest), "--unzip"])
            if (dest / "events.csv").exists():
                return
        except subprocess.CalledProcessError:
            print("[warn] Kaggle API failed, using GitHub mirror")
    tmp = root / "_xg_football"
    _sparse_clone(KAGGLE_MIRROR_URL, tmp, KAGGLE_MIRROR_COMMIT, [], branch="main")
    z = tmp / "Football_events.zip"
    if not z.exists():  # cone mode with no paths keeps root files; be explicit just in case
        _run(["git", "checkout", "-q", "FETCH_HEAD", "--", "Football_events.zip"], cwd=tmp)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(dest)
    shutil.rmtree(tmp, ignore_errors=True)


def download_official(root: Path, splits=("train", "valid", "test")) -> None:
    """Official Labels-v2.json and Labels-caption.json via the SoccerNet package (needs network
    access to the SoccerNet servers; works on Colab)."""
    from SoccerNet.Downloader import SoccerNetDownloader

    dl = SoccerNetDownloader(LocalDirectory=str(root / "soccernet"))
    dl.downloadGames(files=["Labels-v2.json"], split=list(splits))
    try:
        dl.downloadDataTask(task="caption-2023", split=list(splits))
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[warn] caption download failed: {e}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--official", action="store_true", help="also download official SoccerNet labels")
    ap.add_argument("--skip-kaggle", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    download_echoes(root)
    download_mirror(root)
    if not args.skip_kaggle:
        download_kaggle(root)
    if args.official:
        download_official(root)
    print("done ->", root.resolve())


if __name__ == "__main__":
    main()
