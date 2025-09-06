from __future__ import annotations
import os
from pathlib import Path
import zipfile
from typing import Dict, Optional

import pandas as pd
from huggingface_hub import snapshot_download
from datasets import Dataset, DatasetDict, Features, Audio, Value

# ==================== CONFIG ====================
HF_DATASET_REPO = "Rosie-Lab/BERSt"   # destination repo for Arrow/Parquet

# Only mirror the zips (fast) then extract locally
LOCAL_MIRROR_DIR = Path("berst_zip_mirror")
LOCAL_EXTRACT_DIR = Path("berst_extracted")

ZIP_PATHS = {
    "train":      "raw_train-data.zip",
    "validation": "raw_validation-data.zip",
    "test":       "raw_test-data.zip",
}

# Inside each split after extraction:
#   metadata.csv
#   audio/**/final/*.wav
AUDIO_GLOB = "audio/**/final/*.wav"

# Audio feature config
TARGET_SR = 48000
MAX_SHARD_SIZE = "500MB"

# ============ HELPERS ============
def ensure_dirs():
    LOCAL_MIRROR_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

def mirror_only_zips(repo_id: str, zip_relpaths: Dict[str, str]) -> Path:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    allow = list(zip_relpaths.values())
    local_root = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(LOCAL_MIRROR_DIR),
        local_dir_use_symlinks=False,
        allow_patterns=allow,
        max_workers=6,  # keep moderate → fewer 5xx retries
    )
    return Path(local_root)

def extract_zip(mirror_root: Path, split: str, rel_zip: str) -> Optional[Path]:
    zpath = mirror_root / rel_zip
    if not zpath.exists():
        print(f"[skip] {split}: zip not found at {zpath}")
        return None

    out_dir = LOCAL_EXTRACT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unzip into out_dir
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out_dir)

    # Handle "extra folder layer" case
    inner = out_dir / split
    if inner.exists() and inner.is_dir():
        # Move contents up one level
        for child in inner.iterdir():
            target = out_dir / child.name
            if target.exists():
                print(f"[warn] {target} already exists, skipping move of {child}")
            else:
                child.rename(target)
        inner.rmdir()

    print(f"[ok] extracted {zpath} -> {out_dir}")
    return out_dir


def read_split_metadata(split_dir: Path) -> pd.DataFrame:
    meta = split_dir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found — every split must include metadata.csv")
    df = pd.read_csv(meta)

    # Require either 'file_name' or 'path'
    if "file_name" in df.columns and "path" not in df.columns:
        df = df.rename(columns={"file_name": "path"})
    elif "path" not in df.columns:
        raise ValueError(f"{meta} must have 'file_name' or 'path' column with relative file paths")

    # Normalize slashes + strip whitespace from *string* columns (your sample had trailing spaces)
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()

    return df

def infer_feature_types(df: pd.DataFrame) -> Features:
    feats = {"audio": Audio(sampling_rate=TARGET_SR)}
    for c in df.columns:
        if c in ("audio", "path"):
            continue
        s = df[c]
        # Use numeric types when clearly numeric; otherwise keep string (safe for categories like '40-60')
        if pd.api.types.is_integer_dtype(s):
            feats[c] = Value("int64")
        elif pd.api.types.is_float_dtype(s):
            feats[c] = Value("float64")
        else:
            feats[c] = Value("string")
    return Features(feats)

def build_split_dataset(split_dir: Path) -> Dataset:
    df = read_split_metadata(split_dir)

    # Build absolute audio paths, verify files exist
    abs_paths = []
    missing = 0
    for rel in df["path"]:
        ap = (split_dir / rel).resolve()
        if not ap.exists():
            missing += 1
        abs_paths.append(str(ap))
    if missing:
        print(f"   [warn] {missing} rows in metadata reference missing files on disk")

    # Place 'audio' first, keep all other metadata columns exactly as-is
    df = df.copy()
    df["audio"] = abs_paths
    df = df[["audio"] + [c for c in df.columns if c not in ("audio", "path")]]

    feats = {"audio": Audio(sampling_rate=48000)}  # or 16000, your choice
    # add the rest
    for c in df.columns:
        if c == "audio": continue
        feats[c] = Value("string")  # or infer dtype as you had
    ds = Dataset.from_pandas(df, features=Features(feats), preserve_index=False)
    
    return ds

# ==================== MAIN ====================
def main():
    ensure_dirs()

    print("[1/4] Mirroring ZIPs from HF …")
    mirror_root = mirror_only_zips(HF_DATASET_REPO, ZIP_PATHS)

    print("[2/4] Extracting ZIPs …")
    split_dirs: Dict[str, Path] = {}
    for split, rel in ZIP_PATHS.items():
        d = extract_zip(mirror_root, split, rel)
        if d:
            split_dirs[split] = d
    
    if not split_dirs:
        raise SystemExit("No splits extracted. Check ZIP_PATHS and repo contents.")
    
    #split_dirs = {"./berst_extracted/train"), "test": Path("./berst_extracted/test"), "validation": Path("./berst_extracted/validation")}

    print("[3/4] Building DatasetDict from metadata.csv …")
    dsd = DatasetDict()
    for split, sdir in split_dirs.items():
        # basic existence check
        if not any(sdir.glob(AUDIO_GLOB)):
            print(f" [skip] {split}: no files matched '{AUDIO_GLOB}'")
            continue
        print(f" [build] {split}: {sdir}")
        dsd[split] = build_split_dataset(sdir)


    if not dsd:
        raise SystemExit("No datasets built. Verify extracted structure and audio paths in metadata.csv.")

    # (Optional) sanity decode one sample
    try:
        first = next(iter(dsd.keys()))
        _ = dsd[first][0]["audio"]["array"]
        print(f"[sanity] decoded one '{first}' example OK.")
    except Exception as e:
        print("[note] Could not decode a sample (may be fine for huge files):", e)

    print("[4/4] Pushing Arrow/Parquet shards to the Hub …")
    from huggingface_hub import login
    login("YOUR TOKEN")
    dsd.push_to_hub(HF_DATASET_REPO, max_shard_size=MAX_SHARD_SIZE)
    print("[done] Pushed. `load_dataset(\"%s\")` will now be fast & the card will preview." % HF_DATASET_REPO)

if __name__ == "__main__":
    main()