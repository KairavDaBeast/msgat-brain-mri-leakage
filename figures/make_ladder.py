"""Accuracy ladder: the same four classifiers, scored under stricter and stricter evaluation (5-seed means +- SD).
python figures/make_ladder.py  (run from the repo root) -> figures/fig_ladder.png"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c = pd.concat([pd.read_csv(os.path.join(root, "results/v5/C_seeds.csv")), pd.read_csv(os.path.join(root, "results/v5b/C_seeds.csv"))])
d = pd.read_csv(os.path.join(root, "results/v5/D_loso.csv"))

MODELS = [("resnet", "ResNet50", "#2a78d6", "o"), ("mlp", "MLP on features", "#eb6834", "s"),
          ("gat_i", "MS-GAT (graph)", "#1baf7a", "D"), ("knn1", "1-nearest neighbor", "#eda100", "^")]
STEPS = [("random", "Random\nsplit"), ("group", "No copies\nacross split"), ("patient", "No copies or\nshared patients"),
         ("ds3", "New source:\nPMRAM"), ("ds1", "New source:\nNickparvar")]


def stat(model, step):
    if step in ("random", "group", "patient"):
        s = c[(c.split == step) & (c.model == model)].acc * 100
    else:  # leave-one-source-out, scored only on held-out images with no near-copy in the training sources
        s = d[(d.held_out == step) & (d.model == model)].acc_clean * 100
    return s.mean(), s.std()


ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300, facecolor=surface)
ax.set_facecolor(surface)
xs = range(len(STEPS))
ends = []
for i, (m, name, col, mk) in enumerate(MODELS):
    vals = [stat(m, s) for s, _ in STEPS]
    off = (i - 1.5) * 0.06
    ax.errorbar([x + off for x in xs], [v[0] for v in vals], yerr=[v[1] for v in vals], color=col, marker=mk, ms=6,
                lw=2, capsize=0, elinewidth=1.2, label=name, mec=surface, mew=1.5, zorder=3)
    ends.append([vals[-1][0], name, col])
ends.sort()
for k in range(1, len(ends)):  # keep direct labels at least 1.3 points apart
    ends[k][0] = max(ends[k][0], ends[k - 1][0] + 1.3)
for yv, name, col in ends:
    ax.text(len(STEPS) - 1 + 0.18, yv, name, color=ink2, fontsize=7.5, va="center")
ax.set_xticks(list(xs)); ax.set_xticklabels([s[1] for s in STEPS], fontsize=8, color=ink2)
ax.set_ylabel("Test accuracy (%)", fontsize=9, color=ink2)
ax.set_ylim(80, 100); ax.set_xlim(-0.3, len(STEPS) - 1 + 1.25)
ax.axvline(2.5, color=grid, lw=1, ls="--", zorder=1)
ax.text(1.0, 99.2, "same source as training", ha="center", fontsize=7.5, color=ink2)
ax.text(3.5, 99.2, "source never seen in training", ha="center", fontsize=7.5, color=ink2)
ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(grid)
ax.tick_params(colors=ink2, labelsize=8, length=0)
ax.legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2)
fig.tight_layout()
out = os.path.join(root, "figures", "fig_ladder.png")
fig.savefig(out, facecolor=surface)
print("wrote", out)
for s, lab in STEPS:
    print(lab.replace("\n", " "), {m: round(stat(m, s)[0], 2) for m, *_ in MODELS})
