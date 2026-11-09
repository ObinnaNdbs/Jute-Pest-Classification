"""Create the fair train/test split: every copy of a photo stays on one side.

Photo families come from vision/find_photo_families.py. Whole families are assigned
with a stratified grouped split (20% test), so no photo appears on both sides. Images
in a confirmed label conflict are dropped. In test, one image per family is marked
as the representative, so a photo with 30 copies counts once.

Writes fair_split.csv: original, family, fair_fold (0-4, or -1 if dropped), fair_split
(fold 0 = test, the rest train, or dropped), fair_rep.

Usage:
    python make_fair_split.py --family loose
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).parent
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["strict", "loose"], default="loose")
    args = parser.parse_args()

    families = pd.read_csv(PROJECT_DIR / "photo_families.csv", keep_default_na=False)
    meta = pd.read_csv(PROJECT_DIR / "features" / "meta.csv", keep_default_na=False)
    assert (families["original"] == meta["original"]).all(), "photo_families.csv and meta.csv rows differ"

    out = pd.DataFrame({"original": meta["original"], "family": families[f"family_{args.family}"]})
    dropped = (meta["label_conflict"].astype(str) == "True").to_numpy()
    usable = np.where(~dropped)[0]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    out["fair_fold"] = -1
    for fold, (_, test_rows) in enumerate(splitter.split(usable, meta["label"].to_numpy()[usable],
                                                         out["family"].to_numpy()[usable])):
        out.loc[usable[test_rows], "fair_fold"] = fold

    # Fold 0 is the held-out test set; all five folds are used to score every photo once.
    out["fair_split"] = np.where(out["fair_fold"] == 0, "test", "train")
    out.loc[dropped, "fair_split"] = "dropped"

    kept = out[out["fair_split"] != "dropped"]
    assert kept.groupby("family")["fair_fold"].nunique().max() == 1, "a family crosses folds"

    # The representative is a fixed choice that doesn't depend on any model result. One per
    # family and class, because a few families link images filed under different classes.
    reps = kept.assign(cls=meta["class"]).sort_values("original").drop_duplicates(["family", "cls"]).index
    out["fair_rep"] = False
    out.loc[reps, "fair_rep"] = True
    out.to_csv(PROJECT_DIR / "fair_split.csv", index=False)

    summary = meta.assign(fair=out["fair_split"], rep=out["fair_rep"] & (out["fair_split"] == "test"))
    print(f"Families used: {args.family}; train {int((out['fair_split'] == 'train').sum())}, "
          f"test {int((out['fair_split'] == 'test').sum())}, test photos {int(summary['rep'].sum())}, "
          f"photos across all folds {int(out['fair_rep'].sum())}, dropped {int(dropped.sum())}")
    table = pd.DataFrame({
        "train_images": summary[summary["fair"] == "train"].groupby("class").size(),
        "test_photos": summary[summary["rep"]].groupby("class").size(),
        "test_native_photos": summary[summary["rep"] & (summary["source"] == "native_256")].groupby("class").size(),
        "test_resized_photos": summary[summary["rep"] & (summary["source"] == "resized")].groupby("class").size(),
    }).fillna(0).astype(int)
    print(table.to_string())


if __name__ == "__main__":
    main()
