"""Builds splits_v6.csv for the v6 re-test: one row per raw Nickparvar (Training+Testing) and PMRAM image, with
  - copy_group: rotation- and flip-aware near-copy group from leakcheck (<= 4 of 256 bits)
  - patient: Figshare (Cheng et al.) patient ID when the image matches a Figshare slice (NCC >= 0.95, any orientation)
  - group: copy groups joined with same-patient links (use this for leak-free splits)
  - pmram_clean: PMRAM image with no near-copy and no shared patient in Nickparvar (an honest external test)

Usage: python build_splits.py NICKPARVAR_DIR PMRAM_RAW_DIR FIGSHARE_MAT_DIR OUT_CSV
"""
import os, sys, csv
import numpy as np
import cv2, h5py
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "leakcheck"))
import leakcheck as lc


def thumb(a):
    a = cv2.resize(np.asarray(a, dtype=np.float32), (64, 64), interpolation=cv2.INTER_AREA).reshape(-1)
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-8)


def figshare_patients(paths, mat_dir, thr=0.95):
    mats = sorted(f for f in os.listdir(mat_dir) if f[:-4].isdigit() and f.endswith(".mat"))
    T, pid = [], []
    for f in mats:
        with h5py.File(os.path.join(mat_dir, f), "r") as h:
            img = np.array(h["cjdata"]["image"], dtype=np.float32)
            p = "".join(chr(int(c)) for c in np.array(h["cjdata"]["PID"]).reshape(-1) if int(c) > 0).strip()
        for k in range(4):
            for fl in (False, True):
                T.append(thumb(np.rot90(img.T if fl else img, k))); pid.append(p)
    T = np.stack(T); pid = np.array(pid, dtype=object)
    out = np.array([""] * len(paths), dtype=object)
    for s in range(0, len(paths), 512):
        Q = np.stack([thumb(np.array(Image.open(p).convert("L"))) for p in paths[s:s + 512]])
        S = Q @ T.T
        best, arg = S.max(1), S.argmax(1)
        out[s:s + 512] = np.where(best >= thr, pid[arg], "")
    print(f"figshare: {len(mats)} slices, {len(set(pid))} patients; matched {int((out != '').sum())} images", file=sys.stderr)
    return out


def main(nick, pmram, mat_dir, out_csv):
    sets = [("nick_train", os.path.join(nick, "Training")), ("nick_test", os.path.join(nick, "Testing")), ("pmram", pmram)]
    paths, set_of, labels = [], [], []
    for name, root in sets:
        f = lc.find_images(root)
        paths += f; set_of += [name] * len(f); labels += [lc.norm_label(os.path.basename(os.path.dirname(p))) for p in f]
    n = len(paths)
    tnames, bits, ok = lc.hash_all(paths)
    pairs = lc.near_pairs(bits, ok, 4)
    copy_group = lc.groups_from_pairs(n, pairs)
    patient = figshare_patients(paths, mat_dir)
    links = [(i, j, 0, 0) for i, j, _, _ in pairs]
    first = {}
    for i, p in enumerate(patient):
        if p:
            if p in first:
                links.append((first[p], i, 0, 0))
            else:
                first[p] = i
    group = lc.groups_from_pairs(n, links)
    nick_groups = {group[i] for i in range(n) if set_of[i] != "pmram"}
    pmram_clean = [set_of[i] == "pmram" and group[i] not in nick_groups for i in range(n)]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "set", "label", "copy_group", "patient", "group", "pmram_clean"])
        for i in range(n):
            parts = paths[i].replace("\\", "/").split("/")
            key = "/".join(parts[-3:] if set_of[i] != "pmram" else parts[-2:])   # Training/glioma/x.jpg  or  512Glioma/x.jpg
            w.writerow([key, set_of[i], labels[i], copy_group[i], patient[i], group[i], int(pmram_clean[i])])
    pm = [i for i in range(n) if set_of[i] == "pmram"]
    print(f"{n} images; PMRAM clean (no copy or shared patient in Nickparvar): {sum(pmram_clean)}/{len(pm)}", file=sys.stderr)
    te = [i for i in range(n) if set_of[i] == "nick_test"]
    tr_groups = {group[i] for i in range(n) if set_of[i] == "nick_train"}
    print(f"official Nickparvar test images sharing a copy or patient group with Training: "
          f"{sum(group[i] in tr_groups for i in te)}/{len(te)}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:5])
