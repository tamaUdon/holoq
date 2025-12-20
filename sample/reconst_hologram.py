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
    # フレネル回折のインパルス応答
    h = np.zeros((constants.X * 2, constants.Y * 2), dtype=np.int64)
    for n in range(constants.X):
        for m in range(constants.Y):
            idx = m + n  # * constants.X
            dx = (m - constants.X // 2) * constants.pp
            dy = (n - constants.X // 2) * constants.pp
            phase = (dx * dx + dy + dy) * math.pi / (constants.λ * constants.d)
            h[idx][0] = math.cos(phase)
            h[idx][1] = math.sin(phase)
    print(f"h={h}")
    return h


def fresnel_fft(
    points: np.ndarray,
    constants: Constants,
) -> np.ndarray:
    # ゼロパディングありの画像 * FFT --> 出力: F[a]
    # インパルス応答 * FFT --> 出力: F[a]
    # F[a] * F[a] * IFFT --> 出力: ゼロパディングありのμ
    # ゼロパディングを除く --> 出力: 周りこみのないμ

    pad_points = np.pad(points, constants.pad)
    # DEBUG
    # nonpad_points = points
    h = response(constants)

    fa = np.fft.fft(pad_points)
    fb = np.fft.fft(h)
    μ = np.fft.ifft2(np.matmul(fa, fb))
    return μ


def fresnel_propagation(field: np.ndarray, constants: Constants) -> np.ndarray:
    fx = np.fft.fftfreq(constants.X, d=constants.pp)
    fy = np.fft.fftfreq(constants.Y, d=constants.pp)
    fx, fy = np.meshgrid(fx, fy)
    h = np.exp(1j * constants.k * constants.d) * np.exp(
        -1j * math.pi * constants.λ * constants.d * (fx * fx + fy * fy)
    )
    spectrum = np.fft.fft2(field)
    return np.fft.ifft2(spectrum * h)


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
    # recon = fresnel_propagation(hologram.astype(np.complex128), constants)
    recon = fresnel_fft(hologram, constants)

    end = time.time()
    print(f"Cal time: {end - start:.3f} sec")
    show(hologram, recon)


if __name__ == "__main__":
    main()

# TODO
# 0. ホログラフィ入門の、フレネル回折の章を一読する
# 1. fresnel_propagation()の中身を理解する
# 2. ゼロパディングを実装する
# 3. エイリアシングが発生しないzを計算する部分を実装する
# 4. generate_hologram()を写経する (Optional)
# 5. 研究計画スライドを作る
