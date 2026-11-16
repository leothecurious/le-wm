"""Pick the grounding weight from 2-epoch sweep runs.

Among runs whose predicted embedding beats copy-last on validation, take the one
whose pixel share of the encoder-output gradient norm (last-epoch mean) is closest
to 1/2, and report the lowest latent share seen, which should stay above 1/10.
``--weight lam`` (latent-augmented twin) swaps the two terms' roles.

Usage: uv run scripts/pick_beta.py [--weight beta|lam] RUN_DIR [RUN_DIR ...]
RUN_DIR is $STABLEWM_HOME/checkpoints/NAME; metrics.csv is read from
$STABLEWM_HOME/spt/NAME/runs, where scripts/run.sh puts it.
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


def column(values, name):
    """``name``, or the epoch aggregate Lightning logs as ``name_epoch``; None if neither."""
    return values.get(name, values.get(f"{name}_epoch"))


def metrics_path(run_dir):
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
        "share": column(last, f"fit/{weighted}_grad_share"),
        "other_share_min": min(
            v for v in (column(e, f"fit/{other}_grad_share_min") for e in epochs.values()) if v is not None
        ),
        "pred": column(last, "validate/pred_loss"),
        "copy": column(last, "validate/copy_loss"),
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
