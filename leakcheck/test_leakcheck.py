"""Self-check: plant rotated / mirrored / resized copies in a 'test' set and make sure leakcheck finds each one."""
import os, tempfile
import numpy as np
from PIL import Image
import leakcheck


def blob_image(seed, size=128):
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[:size, :size]
    a = np.zeros((size, size))
    for _ in range(6):  # a few soft blobs at random spots -> asymmetric, distinct images
        cy, cx, r = rng.randint(10, size - 10, 2).tolist() + [rng.randint(6, 30)]
        a += rng.rand() * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * r * r))
    return Image.fromarray((255 * a / a.max()).astype(np.uint8))


def test_finds_every_planted_copy():
    with tempfile.TemporaryDirectory() as d:
        tr, te = os.path.join(d, "train", "glioma"), os.path.join(d, "test", "glioma")
        os.makedirs(tr); os.makedirs(te)
        for k in range(20):
            blob_image(k).save(os.path.join(tr, f"{k}.png"))
        src = blob_image(3)
        planted = {
            "rot90": src.transpose(Image.Transpose.ROTATE_90),
            "rot180": src.transpose(Image.Transpose.ROTATE_180),
            "mirror": src.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            "resized": src.resize((256, 256)),
        }
        for name, im in planted.items():
            im.save(os.path.join(te, f"copy_{name}.png"))
        for k in range(100, 105):  # genuinely new images
            blob_image(k).save(os.path.join(te, f"new_{k}.png"))

        _, summary, _ = leakcheck.run([("train", os.path.join(d, "train")), ("test", os.path.join(d, "test"))])
        share = summary["cross_set_share"]["test"]["train"]
        assert abs(share - 4 / 9) < 1e-9, share          # exactly the 4 planted copies out of 9 test images

        _, summary_flips, _ = leakcheck.run([("train", os.path.join(d, "train")), ("test", os.path.join(d, "test"))],
                                            rotations=False)
        assert summary_flips["cross_set_share"]["test"]["train"] < share   # flips-only misses the rotations


if __name__ == "__main__":
    test_finds_every_planted_copy()
    print("leakcheck self-check passed")
