import numpy as np
import tqdm
import time
from constants import Constants
from pointcloud import create_rectangle_points, show


def generate_hologram(points: np.ndarray, constants: Constants) -> np.ndarray:
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        dx = xx - xj * constants.pp
        dy = yy - yj * constants.pp
        r = np.sqrt(dx * dx + dy * dy + zj * zj)
        hologram += np.cos(constants.k * r) / r
    return hologram


def monopolar(points: np.ndarray, constants: Constants):
    # Complex, amplitude and phase-only holograms using bipolar approximationのFig.2を参考に作成
    w1_bits = constants.fractional_bit
    frac_bits = constants.fractional_bit
    scale = 1 << frac_bits
    target_bit = frac_bits - 1

    x = np.arange(constants.X, dtype=np.int64)
    y = np.arange(constants.Y, dtype=np.int64)
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        dx = xx.astype(np.float64) - xj
        dy = yy.astype(np.float64) - yj
        w1 = np.round(dx * dx + dy * dy + zj * zj).astype(np.int64)
        w1 = w1 & ((1 << w1_bits) - 1)
        theta = (constants.pp * constants.pp) / (2.0 * constants.λ * zj)
        w2 = int(round(theta * scale))
        theta = w1 * w2
        t = (theta >> target_bit) & 1

        hologram += t.astype(np.float64)
    return hologram


def monopolar_numpy(points: np.ndarray, constants: Constants):
    # numpy実装版 - 512*512画素で7sec
    x = np.arange(constants.X, dtype=np.float64) * constants.pp
    y = np.arange(constants.Y, dtype=np.float64) * constants.pp
    xx, yy = np.meshgrid(x, y)
    hologram = np.zeros((constants.Y, constants.X), dtype=np.float64)

    for xj, yj, zj in tqdm.tqdm(points):
        hx = xx - xj * constants.pp
        hy = yy - yj * constants.pp
        rho = constants.k / zj
        phase = rho * (hx * hx + hy * hy + zj * zj)
        hologram += np.where(np.cos(phase) >= 0.0, 1.0, -1.0)

    return hologram


def main():
    start = time.time()
    constants = Constants()

    points = create_rectangle_points(constants)  # 四角形 # TODO - 分岐
    hologram = monopolar(points, constants)
    show(hologram)
    print("CGH Calculation completed!")

    end = time.time()
    print(print("Cal time:{} sec".format(end - start)))

    print("Preparing for display...")
    show(hologram)


if __name__ == "__main__":
    main()
