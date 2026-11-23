# leakcheck

Finds copies of images across your train/test splits or across datasets, including copies that were
resized, re-saved, mirrored, flipped, or rotated. One command, plain-text report.

```
pip install pillow imagehash numpy
python leakcheck.py train=data/Training test=data/Testing --out report/
```

Each argument is `NAME=FOLDER`. Images are found recursively and the parent folder name is used as the class label
(`512Glioma`, `glioma_tumor` and `glioma` all count as the same label). Optional flags:

| Flag | Meaning |
|---|---|
| `--threshold 4` | Max differing bits (of 256) for two images to count as copies. 4 is the default. |
| `--flips-only` | Check only the original and two flips, not rotations (faster, misses rotated copies) |
| `--ids patients.csv` | CSV with columns `path,patient`; adds a patient-overlap check between sets |
| `--out folder` | Writes `report.txt`, `images.csv` (with a near-copy `group` id per image) and `pairs.csv` |

To build a split with no copies across it, pass the `group` column of `images.csv` to scikit-learn's
`GroupKFold` or `StratifiedGroupKFold`.

## How it works

Every image gets a 256-bit perceptual hash (pHash) in all 8 orientations (original, 3 rotations, and the mirror of each).
Two images are near-copies when the best-matching orientations differ in at most `--threshold` bits.
Near-copies are joined into groups (connected components), so chains of copies end up in one group.

## Example: three public brain tumor MRI datasets

`results/leakcheck_demo/report.txt` in this repo, run on the Nickparvar Kaggle dataset (its own train and test folders),
BRISC 2025, and the PMRAM raw images (14,705 images, about 80 seconds on a laptop CPU):

| Set | Images | Share with a near-copy in... |
|---|---|---|
| Nickparvar test | 1,600 | Nickparvar train: **66.9%** |
| BRISC 2025 | 6,000 | Nickparvar train: **75.2%** |
| PMRAM | 1,505 | Nickparvar train: **54.8%** |

1,110 of the cross-set copies only match after a 180-degree rotation, which a flips-only check misses.

## Limits

Perceptual hashing finds copies of the *same image*. It does not find different slices of the same patient,
heavy crops, or strong contrast changes. Use `--ids` when patient IDs are known.

## Self-check

```
python test_leakcheck.py
```

plants rotated, mirrored and resized copies among synthetic images and checks that each one is found.
