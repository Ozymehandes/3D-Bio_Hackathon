from pep_utils import load_peptide_data         
from train_test_split import split_by_protein   
from esm_embeddings import get_esm_model, get_esm_embeddings
import pickle, gzip
from pathlib import Path

pos_dict, neg_dict, doubt_dict = load_peptide_data(
    data_csv="DB/NesDB_all_CRM1_with_peptides_train.csv",
    include_nesdoubt=True,      
    include_nodoubt=True,
)


pos_train, pos_val, pos_test, neg_train, neg_val, neg_test, doubt_tr, doubt_val, doubt_te = split_by_protein(
    pos_dict, neg_dict, doubt_dict,
    test_size=0.15,     
    dev_size =0.15,
    random_state=17,
)


OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

split_dicts = {
    "pos_train": pos_train,
    "pos_val":   pos_val,
    "pos_test":  pos_test,
    "neg_train": neg_train,
    "neg_val":   neg_val,
    "neg_test":  neg_test,
}

with gzip.open(OUT_DIR / "peptide_splits.pkl.gz", "wb") as fh:
    pickle.dump(split_dicts, fh, protocol=pickle.HIGHEST_PROTOCOL)

print(f"proteins  train={len({k[0] for k in pos_train}|{k[0] for k in neg_train})}  "
      f"val={len({k[0] for k in pos_val}|{k[0] for k in neg_val})}  "
      f"test={len({k[0] for k in pos_test}|{k[0] for k in neg_test})}")
print(f"segments  train={len(pos_train)+len(neg_train)}  "
      f"val={len(pos_val)+len(neg_val)}  test={len(pos_test)+len(neg_test)}")

# Mirror the split for ZPS segments
import json
from pathlib import Path

# collect the protein IDs that ended up in each bucket
train_prots = {k[0] for k in pos_train} | {k[0] for k in neg_train}
val_prots   = {k[0] for k in pos_val}   | {k[0] for k in neg_val}
test_prots  = {k[0] for k in pos_test}  | {k[0] for k in neg_test}

print(f"train_prots={len(train_prots)}  val_prots={len(val_prots)}  "
      f"test_prots={len(test_prots)}")

# prepare output paths
OUT_DIR = Path("data/zps_split")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_fh  = { "train": (OUT_DIR/"zps_segments_train.jsonl").open("w"),
            "val":   (OUT_DIR/"zps_segments_val.jsonl").open("w"),
            "test":  (OUT_DIR/"zps_segments_test.jsonl").open("w") }

bucket_objs = { "train": [], "val": [], "test": [] }
bucket_ctr  = { "train": 0,  "val": 0,  "test": 0,  "skipped": 0 }

# Output train test and validation segments
with open("data/zps_segments.jsonl") as fh:          # see format in zps.py:
    for line in fh:
        obj = json.loads(line)
        acc = obj["id"]                            # UniProt accession

        if   acc in train_prots:
            bucket = "train"
        elif acc in val_prots:
            bucket = "val"
        elif acc in test_prots:
            bucket = "test"
        else:
            bucket = None       # segment belongs to a protein we ignored(peptide length is too long)
            bucket_ctr["skipped"] += 1

        if bucket:
            out_fh[bucket].write(line)

            bucket_objs[bucket].append(obj)
            bucket_ctr[bucket] += 1

for fh in out_fh.values():
    fh.close()

print("\nZPS-segment rows dispatched:")
for k, v in bucket_ctr.items():
    print(f"  {k:8s}: {v}")


