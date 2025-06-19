import json, numpy as np
from pathlib import Path

IN_DIR  = Path("data/zps_split")          
OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def jsonl_to_npz(split_name):
    """stream *.jsonl → {X, y, keys}.npz"""
    keys, feats, labels = [], [], []
    with open(IN_DIR / f"zps_segments_{split_name}.jsonl") as fh:
        for ln in fh:
            obj = json.loads(ln)

            keys.append(f"{obj['id']} {obj['coords'][0]}-{obj['coords'][1]}")

            feats.append(obj["embedding"])      
            labels.append(obj["label"])         # 0 or 1

    X = np.asarray(feats,  dtype=np.float32)    
    y = np.asarray(labels, dtype=np.int8)       
    k = np.asarray(keys,   dtype=object)        

    np.savez_compressed(OUT_DIR / f"{split_name}.npz", X=X, y=y, keys=k)
    print(f"{split_name:5s} → {X.shape[0]:6,d} rows   saved to {split_name}.npz")

for split in ("train", "val", "test"):
    jsonl_to_npz(split)

