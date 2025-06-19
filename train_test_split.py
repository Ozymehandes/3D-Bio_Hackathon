from collections import defaultdict
from sklearn.model_selection import train_test_split

def split_by_protein(pos_dict, neg_dict, doubt_dict,
                     test_size=0.15, dev_size=0.15, random_state=17):
    """
    Return six dictionaries keyed exactly like the originals:
      pos_train, pos_dev, pos_test,
      neg_train, neg_dev, neg_test,
    and three doubt dicts (same keys).
    """
    prot2keys_pos = defaultdict(list)
    prot2keys_neg = defaultdict(list)

    for key in pos_dict:         # key = (acc, pep_seq)
        prot2keys_pos[key[0]].append(key)
    for key in neg_dict:
        prot2keys_neg[key[0]].append(key)

    # 2) unique list of protein IDs
    proteins = sorted(set(list(prot2keys_pos) + list(prot2keys_neg)))

    # 3) train / temp split, then temp to dev / test
    train_prot, temp_prot = train_test_split(
        proteins, test_size=(dev_size + test_size),
        random_state=random_state, shuffle=True)

    rel_test  = test_size / (dev_size + test_size)
    dev_prot, test_prot = train_test_split(
        temp_prot, test_size=rel_test,
        random_state=random_state, shuffle=True)

    buckets = {"train": set(train_prot),
               "dev":   set(dev_prot),
               "test":  set(test_prot)}

    # 4) allocate dictionaries
    splits = {name: (dict(), dict(), dict())   # pos, neg, doubt
              for name in buckets}

    for bucket, protset in buckets.items():
        pos_out, neg_out, dt_out = splits[bucket]

        for prot in protset:
            for k in prot2keys_pos.get(prot, []):
                pos_out[k] = pos_dict[k]
                dt_out[k]  = doubt_dict[k]
            for k in prot2keys_neg.get(prot, []):
                neg_out[k] = neg_dict[k]
                dt_out[k]  = doubt_dict[k]

    return (splits["train"][0], splits["dev"][0], splits["test"][0],   # positives
            splits["train"][1], splits["dev"][1], splits["test"][1],   # negatives
            splits["train"][2], splits["dev"][2], splits["test"][2])   # doubt