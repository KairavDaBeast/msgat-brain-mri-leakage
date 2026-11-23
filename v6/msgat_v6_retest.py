# %% [markdown]
# # v6: do published 99% recipes survive leak-free testing? (+ DINOv2 as a second backbone)
#
# Inputs (Kaggle "Add Input"): the Nickparvar MRI dataset (Training/Testing folders), the PMRAM dataset, and
# `splits_v6.csv` (built locally by `v6/build_splits.py` with the leakcheck tool + Figshare patient matching).
#
# **Part 1. DINOv2 (frozen) features**, 5 seeds: logistic regression, MLP, inductive GAT, 1-NN on
# random / official / copy-free / copy+patient-free splits, plus an outside test on PMRAM (all images vs. only
# images with no copy or shared patient in Nickparvar).
#
# **Part 2. Published recipes, rebuilt** (1 seed each):
# - R1: CONMEDIA 2025 public notebook (ResNet50 full fine-tune; reported 99.85%, best model picked on the test set)
# - R2: BrainFusion, arXiv 2509.24149 (VGG16, last block fine-tuned; reported 99.86% on a random 80/10/10 split)
# Each runs: as published -> official split with a real validation set -> copy-free -> copy+patient-free,
# and the official and copy+patient-free models are also tested on PMRAM (all vs. clean).
# Output: `/kaggle/working/msgat_v6_results.zip`.

# %%
import os, sys, subprocess, importlib


def ensure(pip_name, module):
    try:
        importlib.import_module(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)


ensure("torch_geometric", "torch_geometric")

import json, time, copy, math, random, traceback, shutil, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings("ignore")

INPUT = os.environ.get("MSGAT_INPUT", "/kaggle/input")
WORK = os.environ.get("MSGAT_WORK", "/kaggle/working")
SMOKE = os.environ.get("MSGAT_SMOKE", "0") == "1"
OUT = os.path.join(WORK, "msgat_v6_results"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = dev.type == "cuda"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]; CI = {c: i for i, c in enumerate(CLASSES)}
CFG = dict(seed=42, dino="facebook/dinov2-base", dino_seeds=[42, 7, 123, 2024, 99],
           r1=dict(epochs=30, patience=5), r2=dict(epochs=50, patience=8),
           workers=min(4, os.cpu_count() or 1), deadline_h=11.0, smoke_n=400)
if SMOKE:
    CFG.update(dino="facebook/dinov2-small", dino_seeds=[42, 7], r1=dict(epochs=1, patience=1),
               r2=dict(epochs=1, patience=1), workers=0)
T0 = time.time()
R = {"config": CFG, "smoke": SMOKE, "errors": {}, "runs": [], "dino": [], "audit": {}}


def log(*a):
    m = f"[{(time.time() - T0) / 60:7.1f} min] " + " ".join(str(x) for x in a)
    print(m, flush=True)
    open(os.path.join(OUT, "run_log.txt"), "a").write(m + "\n")


def save():
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def time_ok(margin_min=0):
    return (time.time() - T0) / 3600 < CFG["deadline_h"] - margin_min / 60


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


log("device", dev, "| smoke" if SMOKE else "| full run")

# %% Data: map splits_v6.csv keys to image files
csv_path = None
img_index = {}
for dp, dns, fns in os.walk(INPUT):
    dns.sort()
    if "augment" in dp.lower() or "brisc" in dp.lower():
        continue
    for f in sorted(fns):
        full = os.path.join(dp, f)
        if f == "splits_v6.csv":
            csv_path = full
        elif f.lower().endswith((".jpg", ".jpeg", ".png")):
            parts = full.replace("\\", "/").split("/")
            for k in ("/".join(parts[-3:]), "/".join(parts[-2:])):
                img_index.setdefault(k, []).append(full)
assert csv_path, "splits_v6.csv not found under the inputs"
S = pd.read_csv(csv_path, keep_default_na=False)
S["path"] = [img_index.get(k, [None])[0] for k in S.key]
dups = [k for k in S.key if len(img_index.get(k, [])) > 1]
assert not dups, f"ambiguous image keys, e.g. {dups[:3]}"
missing = S.path.isna().sum()
log(f"splits_v6.csv: {len(S)} rows, {missing} without an image file")
if SMOKE:
    S = S[S.path.notna()]
    S = S.groupby(["set", "label"]).head(CFG["smoke_n"] // 8).reset_index(drop=True)
assert S.path.notna().all(), "some images listed in splits_v6.csv were not found; check the attached datasets"
S["y"] = S.label.map(CI)
nick = S[S.set != "pmram"].reset_index(drop=True)
pm = S[S.set == "pmram"].reset_index(drop=True)
y = nick.y.values
pm_clean = pm.pmram_clean.astype(int).values == 1
log(f"Nickparvar {len(nick)} | PMRAM {len(pm)} (clean {pm_clean.sum()})")

# %% Splits
def split_off(idx, frac, seed, groups=None):
    idx = np.asarray(idx)
    if groups is None:
        a, b = train_test_split(idx, test_size=frac, random_state=seed, stratify=y[idx])
        return np.sort(a), np.sort(b)
    sg = StratifiedGroupKFold(n_splits=max(2, round(1 / frac)), shuffle=True, random_state=seed)
    a, b = next(sg.split(idx, y[idx], groups[idx]))
    return np.sort(idx[a]), np.sort(idx[b])


def make_splits(seed):
    allidx = np.arange(len(nick))
    off_tr = np.where(nick.set == "nick_train")[0]; off_te = np.where(nick.set == "nick_test")[0]
    sp = {"official_select_on_test": (off_tr, off_te, off_te)}
    tr, va = split_off(off_tr, 0.15, seed); sp["official"] = (tr, va, off_te)
    a, te = split_off(allidx, 0.10, seed); tr, va = split_off(a, 1 / 9, seed); sp["random_80_10_10"] = (tr, va, te)
    a, te = split_off(allidx, 0.20, seed); tr, va = split_off(a, 0.15, seed); sp["random"] = (tr, va, te)
    for name, col in (("copy_free", "copy_group"), ("copy_patient_free", "group")):
        g = nick[col].astype(str).values
        a, te = split_off(allidx, 0.20, seed, g); tr, va = split_off(a, 0.15, seed, g)
        assert not (set(g[te]) & set(g[a])) and not (set(g[tr]) & set(g[va])), name
        sp[name] = (tr, va, te)
    return sp


def audit(tr, va, te):
    """Share of test images that share a copy group / a copy-or-patient group with training+validation."""
    seen_c = set(nick.copy_group.astype(str).values[np.r_[tr, va]]); seen_g = set(nick.group.astype(str).values[np.r_[tr, va]])
    tc = nick.copy_group.astype(str).values[te]; tg = nick.group.astype(str).values[te]
    return {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "test_with_copy_in_train": float(np.mean([c in seen_c for c in tc])),
            "test_with_copy_or_patient_in_train": float(np.mean([g in seen_g for g in tg]))}


SPLITS = make_splits(CFG["seed"])
for k, (tr, va, te) in SPLITS.items():
    R["audit"][k] = audit(tr, va, te) if k != "official_select_on_test" else audit(tr, np.array([], int), te)
    log("split", k, R["audit"][k])
save()

# %% Metrics
def scores(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    labs = sorted(set(yt.tolist()))
    return {"n": int(len(yt)), "acc": float((yt == yp).mean()), "macro_f1": float(f1_score(yt, yp, labels=labs, average="macro", zero_division=0))}


def pm_scores(pred_pm):
    yp = pm.y.values
    return {"pmram_all": scores(yp, pred_pm), "pmram_clean": scores(yp[pm_clean], pred_pm[pm_clean])}

# %% Part 1: DINOv2 frozen features
def dino_features():
    from transformers import AutoModel
    m = AutoModel.from_pretrained(CFG["dino"]).to(dev).eval()
    tf = T.Compose([T.Resize((224, 224)), T.Grayscale(3), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    out = []
    paths = list(nick.path) + list(pm.path)
    with torch.no_grad():
        for s in range(0, len(paths), 64):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s + 64]]).to(dev)
            with torch.autocast(device_type=dev.type, enabled=AMP):
                h = m(pixel_values=x).last_hidden_state.float()
            out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy())
    Fm = np.concatenate(out)
    del m; torch.cuda.empty_cache()
    return Fm[:len(nick)], Fm[len(nick):]


class MLP(nn.Module):
    def __init__(self, d, p=0.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ELU(), nn.Dropout(p),
                                 nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(p), nn.Linear(128, 4))
    def forward(self, x):
        return self.net(x)


def fit_full_batch(model, fwd, ytr, yva, epochs=200, patience=25, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, state, bad = 1e9, None, 0
    for ep in range(epochs if not SMOKE else 3):
        model.train(); opt.zero_grad()
        o_tr, _ = fwd(True); crit(o_tr, ytr).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            _, o_va = fwd(False); vl = crit(o_va, yva).item()
        if vl < best:
            best, state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model


def feature_models(Fn, Fp, tr, va, te, seed):
    """Returns predictions on te (Nickparvar) and on all PMRAM images for each feature model."""
    set_seed(seed)
    sc = StandardScaler().fit(Fn[tr]); Xn, Xp = sc.transform(Fn).astype(np.float32), sc.transform(Fp).astype(np.float32)
    preds = {}
    lr = LogisticRegression(max_iter=3000, C=0.5).fit(Xn[np.r_[tr, va]], y[np.r_[tr, va]])
    preds["logreg"] = (lr.predict(Xn[te]), lr.predict(Xp))
    U = Fn / np.linalg.norm(Fn, axis=1, keepdims=True); Up = Fp / np.linalg.norm(Fp, axis=1, keepdims=True)
    ref = np.r_[tr, va]
    nn1 = NearestNeighbors(n_neighbors=1, metric="cosine").fit(U[ref])
    preds["knn1"] = (y[ref][nn1.kneighbors(U[te])[1][:, 0]], y[ref][nn1.kneighbors(Up)[1][:, 0]])
    xt = torch.tensor(Xn, device=dev); xp = torch.tensor(Xp, device=dev); yt = torch.tensor(y, device=dev)
    mlp = MLP(Xn.shape[1]).to(dev)
    mlp = fit_full_batch(mlp, lambda train: (mlp(xt[tr]), mlp(xt[va])), yt[tr], yt[va])
    with torch.no_grad():
        preds["mlp"] = (mlp(xt[te]).argmax(1).cpu().numpy(), mlp(xp).argmax(1).cpu().numpy())
    preds["gat_inductive"] = gat_inductive(Xn, Xp, tr, va, te, seed)
    return preds


def gat_inductive(Xn, Xp, tr, va, te, seed, k=8):
    """Same idea as MS-GAT's inductive graph: train/val nodes link among themselves; every test or PMRAM node
    receives edges from its k nearest train/val nodes and sends none."""
    from torch_geometric.nn import GATConv
    set_seed(seed)
    ref = np.r_[tr, va]; A = len(ref)
    X = np.concatenate([Xn[ref], Xn[te], Xp]); n = len(X)
    Un = X / np.linalg.norm(X, axis=1, keepdims=True)
    nnr = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Un[:A])
    _, ia = nnr.kneighbors(Un[:A]); ia = np.stack([r[r != i][:k] for i, r in enumerate(ia)])
    _, iq = nnr.kneighbors(Un[A:], n_neighbors=k)
    src = np.concatenate([ia.reshape(-1), iq.reshape(-1)])
    dst = np.concatenate([np.repeat(np.arange(A), k), np.repeat(np.arange(A, n), k)])
    ei = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=dev)
    ei_tr = ei[:, :A * k]

    class G(nn.Module):
        def __init__(s, d):
            super().__init__()
            s.c1 = GATConv(d, 256, heads=4, dropout=0.4); s.b1 = nn.BatchNorm1d(1024)
            s.c2 = GATConv(1024, 128, heads=1, dropout=0.4); s.b2 = nn.BatchNorm1d(128); s.out = nn.Linear(128, 4)
        def forward(s, x, e):
            x = F.dropout(F.elu(s.b1(s.c1(x, e))), 0.4, s.training)
            return s.out(F.elu(s.b2(s.c2(x, e))))
    x_all = torch.tensor(X, device=dev); yr = torch.tensor(y[ref], device=dev)
    g = G(X.shape[1]).to(dev)
    ntr = len(tr)
    g = fit_full_batch(g, lambda train: (g(x_all[:A], ei_tr)[:ntr], g(x_all[:A], ei_tr)[ntr:A]), yr[:ntr], yr[ntr:A])
    with torch.no_grad():
        p = g(x_all, ei).argmax(1).cpu().numpy()
    return p[A:A + len(te)], p[A + len(te):]


def part1():
    Fn, Fp = dino_features()
    log("DINOv2 features", Fn.shape)
    for seed in CFG["dino_seeds"]:
        sp = make_splits(seed)
        for name in ("random", "official", "copy_free", "copy_patient_free"):
            tr, va, te = sp[name]
            for m, (pte, ppm) in feature_models(Fn, Fp, tr, va, te, seed).items():
                row = {"backbone": CFG["dino"], "seed": seed, "split": name, "model": m, **scores(y[te], pte),
                       **{f"{k}_{kk}": vv for k, v in pm_scores(ppm).items() for kk, vv in v.items()}}
                R["dino"].append(row)
            log(f"[dino seed {seed}] {name}: " + " | ".join(f"{r['model']} {r['acc']:.4f}" for r in R["dino"][-4:]))
            pd.DataFrame(R["dino"]).to_csv(os.path.join(OUT, "part1_dino.csv"), index=False); save()

# %% Part 2: published recipes
class Imgs(Dataset):
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = list(paths), list(labels), tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])


def recipe(name):
    if name == "R1_conmedia_resnet50":
        norm = T.Normalize([0.5] * 3, [0.5] * 3)
        tf_tr = T.Compose([T.Grayscale(3), T.RandomResizedCrop(224, scale=(0.8, 1.0)), T.RandomRotation(15),
                           T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2), T.ToTensor(), norm])
        tf_ev = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.ToTensor(), norm])
        m = tvm.resnet50(weights=None if SMOKE else "IMAGENET1K_V1"); m.fc = nn.Linear(2048, 4)
        params = m.parameters()
        return dict(model=m, tf_tr=tf_tr, tf_ev=tf_ev, opt=torch.optim.Adam(params, lr=1e-5), bs=16,
                    sampler=True, select="macro_f1", ls=0.1, **CFG["r1"])
    if name == "R2_brainfusion_vgg16":
        # paper: resize 224, scale to [0,1], shear 30%, zoom 30%, h+v flips; VGG16 ImageNet, last 5 Keras layers
        # (block5) trainable, custom head (not specified; we use GAP-256-dropout 0.5), Adam 1e-5, ReduceLROnPlateau,
        # 100 epochs + early stopping (patience not specified; we cap at 50 epochs, patience 8)
        tf_tr = T.Compose([T.Resize((224, 224)), T.RandomAffine(degrees=0, shear=17, scale=(0.7, 1.3)),
                           T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor()])
        tf_ev = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        vgg = tvm.vgg16(weights=None if SMOKE else "IMAGENET1K_V1")
        for i, p in enumerate(vgg.features.parameters()):
            p.requires_grad = False
        for layer in list(vgg.features)[24:]:
            for p in layer.parameters():
                p.requires_grad = True
        m = nn.Sequential(vgg.features, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 4))
        return dict(model=m, tf_tr=tf_tr, tf_ev=tf_ev, opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-5),
                    bs=32, sampler=False, select="val_loss", ls=0.0, **CFG["r2"])
    raise ValueError(name)


@torch.no_grad()
def predict(m, paths, tf):
    m.eval(); out = []
    for x, _ in DataLoader(Imgs(paths, [0] * len(paths), tf), batch_size=64, num_workers=CFG["workers"]):
        with torch.autocast(device_type=dev.type, enabled=AMP):
            out.append(m(x.to(dev)).float().cpu())
    return torch.cat(out)


def run_recipe(rname, split_name, tr, va, te):
    set_seed(CFG["seed"])
    rc = recipe(rname); m = rc["model"].to(dev)
    sel = te if split_name == "official_select_on_test" else va
    ytr = y[tr]
    if rc["sampler"]:
        w = 1.0 / np.bincount(ytr, minlength=4)[ytr]
        dl = DataLoader(Imgs(nick.path.values[tr], ytr, rc["tf_tr"]), batch_size=rc["bs"], num_workers=CFG["workers"],
                        sampler=WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(w), replacement=True))
    else:
        dl = DataLoader(Imgs(nick.path.values[tr], ytr, rc["tf_tr"]), batch_size=rc["bs"], shuffle=True, num_workers=CFG["workers"])
    crit = nn.CrossEntropyLoss(label_smoothing=rc["ls"])
    mode = "max" if rc["select"] == "macro_f1" else "min"
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(rc["opt"], mode=mode, factor=0.2, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, state, bad, hist = (-1 if mode == "max" else 1e9), None, 0, []
    for ep in range(rc["epochs"]):
        m.train()
        for x, yy in dl:
            x, yy = x.to(dev), yy.to(dev)
            rc["opt"].zero_grad()
            with torch.autocast(device_type=dev.type, enabled=AMP):
                loss = crit(m(x), yy)
            scaler.scale(loss).backward(); scaler.step(rc["opt"]); scaler.update()
        lo = predict(m, nick.path.values[sel], rc["tf_ev"])
        v = f1_score(y[sel], lo.argmax(1).numpy(), average="macro") if mode == "max" else F.cross_entropy(lo, torch.tensor(y[sel])).item()
        sched.step(v); hist.append(round(float(v), 5))
        if (v > best) if mode == "max" else (v < best):
            best, state, bad = v, copy.deepcopy(m.state_dict()), 0
        else:
            bad += 1
            if bad >= rc["patience"]:
                break
    m.load_state_dict(state)
    pte = predict(m, nick.path.values[te], rc["tf_ev"]).argmax(1).numpy()
    row = {"recipe": rname, "split": split_name, "epochs_run": len(hist), "selection_curve": hist,
           **R["audit"].get(split_name, {}), **scores(y[te], pte)}
    if split_name in ("official", "copy_patient_free"):
        ppm = predict(m, pm.path.values, rc["tf_ev"]).argmax(1).numpy()
        row.update({f"{k}_{kk}": vv for k, v in pm_scores(ppm).items() for kk, vv in v.items()})
    del m; torch.cuda.empty_cache()
    return row


def part2():
    plan = [("R1_conmedia_resnet50", s) for s in ("official_select_on_test", "official", "copy_free", "copy_patient_free")] + \
           [("R2_brainfusion_vgg16", s) for s in ("random_80_10_10", "official", "copy_free", "copy_patient_free")]
    for rname, sname in plan:
        if not time_ok(margin_min=60):
            log("deadline near; skipping", rname, sname); continue
        try:
            row = run_recipe(rname, sname, *SPLITS[sname])
            R["runs"].append(row)
            log(f"[{rname} | {sname}] acc {row['acc']:.4f} f1 {row['macro_f1']:.4f}" +
                (f" | PMRAM all {row['pmram_all_acc']:.4f} clean {row['pmram_clean_acc']:.4f}" if "pmram_all_acc" in row else ""))
        except Exception as e:
            R["errors"][f"{rname}|{sname}"] = traceback.format_exc(); log("FAILED", rname, sname, e)
        pd.DataFrame([{k: v for k, v in r.items() if k != "selection_curve"} for r in R["runs"]]).to_csv(os.path.join(OUT, "part2_recipes.csv"), index=False)
        save()

# %% Run
for nm, fn in (("part1_dino", part1), ("part2_recipes", part2)):
    try:
        log("===== START", nm); fn(); log("===== DONE", nm)
    except Exception as e:
        R["errors"][nm] = traceback.format_exc(); log("FAILED", nm, e)
    save()
R["total_minutes"] = round((time.time() - T0) / 60, 1); save()
shutil.make_archive(os.path.join(WORK, "msgat_v6_results"), "zip", OUT)
log("saved msgat_v6_results.zip | errors:", list(R["errors"]) or "none")
