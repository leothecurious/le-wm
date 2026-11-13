"""Choose the grounding weight beta from the 2-epoch sweep runs.

Among the runs whose predicted embedding beats copying the last context
embedding on validation, pick the beta whose pixel term's share of the
gradient norm at the encoder output, averaged over the last epoch, is closest
to one half. Reports the lowest latent-term share seen in training, which must
stay above one tenth.

Usage: python scripts/pick_beta.py [--weight beta|lam] RUN_DIR [RUN_DIR ...]
With --weight lam, for the latent-augmented twin, the roles of the two terms are swapped:
the pick is the lam whose latent share is closest to one half and the pixel share is reported.
Each RUN_DIR is a run's checkpoint directory ($STABLEWM_HOME/checkpoints/NAME) with its
config.yaml. Its metrics.csv is read from RUN_DIR, or else from the run's stable-pretraining
cache $STABLEWM_HOME/spt/NAME/runs, where scripts/run.sh puts it.
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

TARGET_SHARE = 0.5
MIN_OTHER_SHARE = 0.1


def epoch_metrics(path):
    """Last logged value of every column in each epoch: {epoch: {column: float}}."""
    epochs = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("epoch"):
                continue
            values = epochs.setdefault(int(row["epoch"]), {})
            for key, text in row.items():
                if key != "epoch" and text not in ("", None):
                    values[key] = float(text)
    return epochs


PREFIXES = {"fit": ("fit", "train"), "validate": ("validate", "val")}  # Lightning stage names, or the toy's


def column(values, stage, name):
    for prefix in PREFIXES[stage]:
        for key in (f"{prefix}/{name}", f"{prefix}/{name}_epoch"):
            if key in values:
                return values[key]
    raise KeyError(f"{stage}/{name} is not in the metrics CSV")


def has(values, stage, name):
    try:
        column(values, stage, name)
    except KeyError:
        return False
    return True


def metrics_path(run_dir):
    if (run_dir / "metrics.csv").exists():
        return run_dir / "metrics.csv"
    (path,) = (run_dir.parent.parent / "spt" / run_dir.name).glob("runs/*/*/*/metrics.csv")
    return path


def summarize(run_dir, weight):
    """Per-run numbers: the weight, the share of the weighted term, the lowest share of the other term."""
    run_dir = Path(run_dir)
    value = yaml.safe_load((run_dir / "config.yaml").read_text())["loss"][weight]
    epochs = epoch_metrics(metrics_path(run_dir))
    last = epochs[max(epochs)]
    weighted, other = ("pixel", "latent") if weight == "beta" else ("latent", "pixel")
    return {
        "weight": value,
        "share": column(last, "fit", f"{weighted}_grad_share"),
        "other_share_min": min(
            column(v, "fit", f"{other}_grad_share_min") for v in epochs.values() if has(v, "fit", f"{other}_grad_share_min")
        ),
        "pred": column(last, "validate", "pred_loss"),
        "copy": column(last, "validate", "copy_loss"),
    }


def pick(rows):
    """The row whose share is closest to one half among those beating copy-last, else None."""
    eligible = [r for r in rows if r["pred"] < r["copy"]]
    return min(eligible, key=lambda r: abs(r["share"] - TARGET_SHARE), default=None)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--weight", choices=("beta", "lam"), default="beta")
    args = parser.parse_args(argv)
    rows = sorted((summarize(d, args.weight) for d in args.run_dirs), key=lambda r: r["weight"])
    print(f"{args.weight:>6} {'share':>10} {'other_share_min':>16} {'val_pred':>10} {'val_copy':>10}")
    for r in rows:
        print(f"{r['weight']:>6g} {r['share']:>10.4g} {r['other_share_min']:>16.4g} {r['pred']:>10.4g} {r['copy']:>10.4g}")
    chosen = pick(rows)
    if chosen is None:
        print(f"no {args.weight} beats copy-last on validation")
        return 1
    print(f"{args.weight} = {chosen['weight']:g}; other term's share minimum {chosen['other_share_min']:.4g}"
          + ("" if chosen["other_share_min"] > MIN_OTHER_SHARE else f" (not above {MIN_OTHER_SHARE})"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
