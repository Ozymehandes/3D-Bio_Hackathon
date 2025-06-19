import re, json, pickle, numpy as np, pandas as pd, tqdm
from pathlib import Path
from esm_embeddings import get_esm_model, get_esm_embeddings
from zps import get_protein_segments, get_protein_segment_embeddings
from pep_utils import load_peptide_data, _extract_uniprot                 


CSV_PATH   = "DB/NesDB_all_CRM1_with_peptides_train.csv"
CACHE_DIR  = Path("cache/per_residue")          # one .npy per protein
CACHE_DIR.mkdir(parents=True, exist_ok=True)
JSONL_OUT  = "data/zps_segments.jsonl"

EMB_SIZE   = 1280                                
EMB_LAYER  = 33
MAX_BKPS   = 10                               # ≤ MAX_BKPS break-points / 100 aa


def _parse_ranges(rng_str):
    """
    Return a list of (start,end) tuples, 0-based, end-exclusive.
    To get range of peptides.
    Accepts:
        '405-413'
        '88-95; 101-112'
        '[(400, 415)]'
    """
    matches = re.findall(r"(\d+)\D+?(\d+)", str(rng_str))
    return [(int(a)-1, int(b)) for a, b in matches]

def load_full_sequences(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path).dropna(subset=["Sequence", "Merged ranges"])
    seqs, truth = {}, {}
    for _, row in df.iterrows():
        acc = _extract_uniprot(row)
        seqs[acc]  = row["Sequence"]
        truth[acc] = _parse_ranges(row["Merged ranges"])
    return seqs, truth

def has_nes_overlap(s, e, nes_ranges, mode="any", thresh=0.5):
    for ns, ne in nes_ranges:
        overlap = max(0, min(e, ne) - max(s, ns))
        if mode == "contains" and s <= ns and e >= ne:
            return True
        if mode == "any" and overlap > 0:
            return True
        if mode == "ratio" and overlap / (ne - ns) >= thresh:
            return True
    return False

# embed every protein once (loads cache if present)
print("loading full-length sequences …")
seqs, truth = load_full_sequences()  
with open("data/protein_sequences.pkl", "wb") as fh:
    pickle.dump(seqs, fh)
        
print("loading ESM-2 model …")
model, alphabet, bconv, device = get_esm_model(embedding_size=EMB_SIZE)

prot_emb = {}                                    
for acc, seq in tqdm.tqdm(seqs.items(), desc="embedding proteins"):
    npy_path = CACHE_DIR / f"{acc}.npy"
    if npy_path.exists():
        prot_emb[acc] = np.load(npy_path)
        continue
    vec = get_esm_embeddings([(acc, seq)], model, alphabet,
                             bconv, device,
                             embedding_layer=EMB_LAYER,
                             sequence_embedding=False)[0]  
    np.save(npy_path, vec.astype("float32"))
    prot_emb[acc] = vec.astype("float32")

print(f"per-residue cache ready ({len(prot_emb)} proteins)")

# ZPS segmentation, segment pooling and labeling
print("running ZPS change-point detection …")
segments = get_protein_segments(prot_emb, max_bkps_per100aa=MAX_BKPS)

print("pooling segments and writing JSONL …")
with open(JSONL_OUT, "w") as fh:
    for acc, seg_list in segments.items():
        for s, e in seg_list:
            vec = prot_emb[acc][s:e].mean(0)               
            threshold = 0.5
            hit = has_nes_overlap(s, e, truth[acc], mode="ratio", thresh=threshold)
            obj = {"id": acc,
                   "coords": [int(s), int(e)],
                   "label": int(hit),
                   "embedding": vec.tolist()}
            fh.write(json.dumps(obj)+"\n")

print("finished:", JSONL_OUT, "contains",
      sum(1 for _ in open(JSONL_OUT)), "candidate segments")

protein_segment_embeddings = get_protein_segment_embeddings(
    prot_emb, segments)

# Save the segments and embeddings to files
import pickle
with open("data/protein_segments.pkl", "wb") as fh:
    pickle.dump(segments, fh)

with open("data/protein_segment_embeddings.pkl", "wb") as fh:
    pickle.dump(protein_segment_embeddings, fh)

