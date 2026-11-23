"""leakcheck: find copies and near-copies across image folders (train/test splits or whole datasets).

Catches what exact-file or exact-hash checks miss: copies that were resized, re-saved, mirrored,
flipped or rotated by 90/180/270 degrees. Optionally checks patient overlap from an ID table.

Usage:
    python leakcheck.py train=path/to/Training test=path/to/Testing
    python leakcheck.py nickparvar=MRI/ brisc=BRISC/ pmram=PMRAM/ --threshold 4 --out report/
    python leakcheck.py train=... test=... --ids patient_ids.csv     # csv columns: path,patient

Each argument is NAME=FOLDER. Images are found recursively; the parent folder name is used as the label.
Outputs (in --out): report.txt, images.csv (one row per image, with its near-copy group id),
pairs.csv (every near-copy pair across sets, with distance, transform and labels).
Use the group column with GroupKFold / StratifiedGroupKFold to build splits with no copies across them.
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageOps
import imagehash

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# the 8 symmetries of a square: identity, 3 rotations, and the mirror image of each
TRANSFORMS = {
    "same": lambda im: im,
    "rot90": lambda im: im.transpose(Image.Transpose.ROTATE_90),
    "rot180": lambda im: im.transpose(Image.Transpose.ROTATE_180),
    "rot270": lambda im: im.transpose(Image.Transpose.ROTATE_270),
    "mirror": ImageOps.mirror,
    "flip": ImageOps.flip,
    "transpose": lambda im: im.transpose(Image.Transpose.TRANSPOSE),
    "transverse": lambda im: im.transpose(Image.Transpose.TRANSVERSE),
}


NO_TUMOR = {"notumor", "notumour", "normal", "healthy", "no", "nontumor"}


def norm_label(folder):
    """'512Glioma', 'glioma_tumor', 'No Tumor' -> 'glioma', 'glioma', 'notumor' so labels compare across datasets."""
    s = "".join(ch for ch in folder.lower() if ch.isalpha())
    s = s.replace("tumor", "") if s not in NO_TUMOR else s
    return "notumor" if s in NO_TUMOR or s == "" else s


def find_images(root):
    out = []
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for f in sorted(fns):
            if f.lower().endswith(EXTS):
                out.append(os.path.join(dp, f))
    return out


def hash_bits(im, hash_size=16):
    return np.asarray(imagehash.phash(im, hash_size=hash_size).hash, dtype=np.uint8).reshape(-1)


def hash_all(paths, hash_size=16, rotations=True):
    """Returns bits[t, i, :] for every transform t (index 0 = original) and image i; unreadable images get None."""
    names = list(TRANSFORMS) if rotations else ["same", "mirror", "flip"]
    n_bits = hash_size * hash_size
    bits = np.zeros((len(names), len(paths), n_bits), np.uint8)
    ok = np.ones(len(paths), bool)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("L")
        except Exception:
            ok[i] = False
            continue
        for t, name in enumerate(names):
            bits[t, i] = hash_bits(TRANSFORMS[name](im), hash_size)
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i + 1}/{len(paths)}", file=sys.stderr, flush=True)
    return names, bits, ok


def near_pairs(bits, ok, threshold, chunk=1024):
    """All pairs (i < j) whose hashes differ in <= threshold bits under the best of the transforms.
    Returns rows of (i, j, distance, transform_index)."""
    O = bits[0].astype(np.float32)
    ones_O = O.sum(1)
    n = len(O)
    out = []
    for s in range(0, n, chunk):
        best = np.full((min(chunk, n - s), n), 1e9, np.float32)
        arg = np.zeros_like(best, dtype=np.int8)
        for t in range(bits.shape[0]):
            Q = bits[t, s:s + chunk].astype(np.float32)
            D = Q.sum(1, keepdims=True) + ones_O[None, :] - 2.0 * (Q @ O.T)   # Hamming distance
            better = D < best
            best = np.where(better, D, best)
            arg = np.where(better, t, arg)
        rows, cols = np.nonzero(best <= threshold + 0.5)
        for r, c in zip(rows, cols):
            i = s + r
            if i < c and ok[i] and ok[c]:
                out.append((i, int(c), int(round(best[r, c])), int(arg[r, c])))
    return out


def groups_from_pairs(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, _, _ in pairs:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(n)]


def run(sets, threshold=4, hash_size=16, rotations=True, ids_csv=None, out_dir=None):
    paths, set_of, labels = [], [], []
    for name, root in sets:
        found = find_images(root)
        if not found:
            raise SystemExit(f"no images found in {root}")
        paths += found
        set_of += [name] * len(found)
        labels += [norm_label(os.path.basename(os.path.dirname(p))) for p in found]
    n = len(paths)
    print(f"hashing {n} images ({'8 orientations' if rotations else 'original + 2 flips'})...", file=sys.stderr)
    tnames, bits, ok = hash_all(paths, hash_size, rotations)
    pairs = near_pairs(bits, ok, threshold)
    grp = groups_from_pairs(n, pairs)

    pid = {}
    if ids_csv:
        with open(ids_csv, newline="") as f:
            for row in csv.DictReader(f):
                pid[os.path.normcase(os.path.abspath(row["path"]))] = row["patient"]
    pid_of = [pid.get(os.path.normcase(os.path.abspath(p)), "") for p in paths]

    names = [s for s, _ in sets]
    # for every image: does it have a near-copy in any OTHER set, and how close is the closest?
    has_copy_in = defaultdict(set)
    exact_in = defaultdict(set)
    for i, j, d, t in pairs:
        if set_of[i] != set_of[j]:
            has_copy_in[i].add(set_of[j]); has_copy_in[j].add(set_of[i])
            if d == 0:
                exact_in[i].add(set_of[j]); exact_in[j].add(set_of[i])
    within = Counter(set_of[i] for i, j, d, t in pairs if set_of[i] == set_of[j])
    label_conflicts = [(i, j) for i, j, d, t in pairs if labels[i] != labels[j]]
    by_transform = Counter(tnames[t] for i, j, d, t in pairs if set_of[i] != set_of[j])

    lines = [f"leakcheck report: {n} images, threshold {threshold} of {hash_size * hash_size} bits, "
             f"{'rotations + flips' if rotations else 'flips only'}", ""]
    unreadable = int((~ok).sum())
    if unreadable:
        lines.append(f"unreadable images: {unreadable}")
    sizes = Counter(grp)
    lines.append(f"near-copy groups: {len(sizes)} (images that sit in a group of 2+: {sum(v for v in sizes.values() if v > 1)}, "
                 f"largest group: {max(sizes.values())})")
    lines.append(f"pairs with different labels (likely label noise or mixed classes): {len(label_conflicts)}")
    lines.append("")
    lines.append("Overlap between sets (share of each set's images with a near-copy in the other set):")
    for a in names:
        ia = [i for i in range(n) if set_of[i] == a]
        cells = []
        for b in names:
            if a == b:
                continue
            k = sum(1 for i in ia if b in has_copy_in[i])
            e = sum(1 for i in ia if b in exact_in[i])
            cells.append(f"in {b}: {k}/{len(ia)} ({100 * k / len(ia):.1f}%), identical hash {e}")
        lines.append(f"  {a} ({len(ia)} images, {within.get(a, 0)} near-copy pairs inside it): " + "; ".join(cells))
    if by_transform:
        lines.append("  cross-set copies by best-matching orientation: " + ", ".join(f"{k} {v}" for k, v in by_transform.most_common()))
    if pid:
        lines.append("")
        lines.append("Patient overlap between sets:")
        for a in names:
            pa = {pid_of[i] for i in range(n) if set_of[i] == a and pid_of[i]}
            for b in names:
                if a < b:
                    pb = {pid_of[i] for i in range(n) if set_of[i] == b and pid_of[i]}
                    lines.append(f"  {a} and {b}: {len(pa & pb)} shared patients ({len(pa)} in {a}, {len(pb)} in {b})")
    report = "\n".join(lines)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "report.txt"), "w") as f:
            f.write(report + "\n")
        with open(os.path.join(out_dir, "images.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path", "set", "label", "group", "patient", "copy_in_other_sets"])
            for i in range(n):
                w.writerow([paths[i], set_of[i], labels[i], grp[i], pid_of[i], ";".join(sorted(has_copy_in[i]))])
        with open(os.path.join(out_dir, "pairs.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["path_a", "set_a", "label_a", "path_b", "set_b", "label_b", "distance", "orientation"])
            for i, j, d, t in pairs:
                w.writerow([paths[i], set_of[i], labels[i], paths[j], set_of[j], labels[j], d, tnames[t]])
    summary = {"n": n, "pairs": len(pairs), "groups": len(sizes), "label_conflicts": len(label_conflicts),
               "cross_set_share": {a: {b: sum(1 for i in range(n) if set_of[i] == a and b in has_copy_in[i]) /
                                       max(1, set_of.count(a)) for b in names if b != a} for a in names},
               "by_transform": dict(by_transform)}
    return report, summary, grp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="NAME=FOLDER, one per split or dataset")
    ap.add_argument("--threshold", type=int, default=4, help="max differing hash bits to call two images copies (default 4 of 256)")
    ap.add_argument("--hash-size", type=int, default=16, help="pHash grid size; bits = size^2 (default 16 -> 256 bits)")
    ap.add_argument("--flips-only", action="store_true", help="check original + 2 flips only, not rotations")
    ap.add_argument("--ids", help="optional CSV with columns path,patient for a patient-overlap check")
    ap.add_argument("--out", help="folder for report.txt, images.csv, pairs.csv")
    a = ap.parse_args(argv)
    sets = []
    for s in a.sets:
        if "=" not in s:
            ap.error(f"expected NAME=FOLDER, got {s!r}")
        name, root = s.split("=", 1)
        sets.append((name, root))
    report, summary, _ = run(sets, a.threshold, a.hash_size, not a.flips_only, a.ids, a.out)
    print(report)
    return summary


if __name__ == "__main__":
    main()
