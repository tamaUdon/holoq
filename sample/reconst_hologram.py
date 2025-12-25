import math
import time

import matplotlib.pyplot as plt
import numpy as np

from constants import Constants
from pointcloud import (
    create_rectangle_points,
    generate_hologram,
)


def response(constants: Constants):
    size_x = constants.X * 2
    size_y = constants.Y * 2
    x = (np.arange(size_x, dtype=np.float64) - size_x / 2) * constants.pp
    y = (np.arange(size_y, dtype=np.float64) - size_y / 2) * constants.pp
    dx, dy = np.meshgrid(x, y)

    phase = (math.pi / (constants.λ * constants.d)) * (dx * dx + dy * dy)
    h = np.exp(1j * phase)
    return np.fft.fft2(np.fft.fftshift(h))


def fresnel_fft(
    points: np.ndarray,
    constants: Constants,
) -> np.ndarray:
    # ゼロパディングありの画像 * FFT --> 出力: F[a]
    # インパルス応答 * FFT --> 出力: F[b]
    # F[a] * F[b] * IFFT --> 出力: ゼロパディングありのμ
    # ゼロパディングを除く --> 出力: まわりこみのないμ

    pad_points = np.pad(points, constants.pad)
    fa = np.fft.fft2(pad_points)
    fb = response(constants)
    μ = np.fft.ifft2(fa * fb)
    return μ


def show(hologram: np.ndarray, recon: np.ndarray) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(hologram, cmap="gray")
    ax[0].set_title("Hologram")
    ax[0].axis("off")

    intensity = np.abs(recon) ** 2
    ax[1].imshow(intensity, cmap="gray")
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    start = time.time()
    constants = Constants()

    points = create_rectangle_points(constants)
    hologram = generate_hologram(points, constants)
    recon = fresnel_fft(hologram.astype(np.complex128), constants)

    end = time.time()
    print(f"Cal time: {end - start:.3f} sec")
    show(hologram, recon)


if __name__ == "__main__":
    main()

# TODO
# 0. ゼロパディングを除く実装
# 1. エイリアシングが発生しないzを計算する部分を実装する
