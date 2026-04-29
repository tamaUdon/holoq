from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2


def create_single_point(x, y, z) -> np.ndarray:
    """
    デバッグ用 1点の点群を作成する関数

    - X*Yの中心に物体点 (1点) がある想定

    :param constants: 定数クラスのオブジェクト
    :type constants: ClassicalConstants
    :return: デバッグ用の物体点 (1点)
    :rtype: np.ndarray
    """
    x0 = x / 2
    y0 = y / 2
    z0 = z  # 物体点までの距離

    return np.array([[x0, y0, z0]], dtype=float)


def load_image(path: str) -> np.ndarray:
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is not None:
        return img_gray / 255
    raise IOError  # 0-index


def create_image():
    xx = np.arange(-256, 256)  # 画像の中心を(0,0)とする
    yy = np.arange(-256, 256)
    np.meshgrid(xx, yy, indexing="ij")


def a1(intense: float) -> np.ndarray:
    # intense...伝播元の画像のI_aの各ピクセルの光波の強度分布
    # TODO - 1になるよう正規化する
    return np.sqrt(intense)


def p1(phase: float) -> float:
    # phase...伝播元の画像のI_aの各ピクセルの光波の位相分布
    return phase / 256


def h(
    z: float, λ: float, r: np.ndarray, W: int, H: int, pp: float
) -> np.ndarray:
    # 角スペクトル法
    # z ... z21
    # r...r21

    fx = np.fft.fftfreq(W, d=pp)
    fy = np.fft.fftfreq(H, d=pp)
    Fx, Fy = np.meshgrid(fx, fy)

    cond = (fx**2 + fy**2) <= (1 / λ**2)
    func = z * np.sqrt((1 / λ**2) - Fx**2 - Fy**2)
    p = np.where(cond, func, 0)
    return np.exp(1j * 2 * np.pi * p)


def show_twin(hologram: np.ndarray, recon: np.ndarray) -> None:
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


def main():
    W: int = 512
    H: int = 512
    d = 180e-3
    pp = 10e-6
    λ = 633e-9

    # 座標用
    x = np.arange(-W, W) * pp
    y = np.arange(-H, H) * pp  # ゼロパディングに備えて2倍の座標幅を用意する

    # u1の作成
    # u1 = np.zeros((H, W))
    # u1[H // 2, W // 2] = 1  # 真ん中だけ(1,1)にする
    u1 = load_image("./sample/wavefront/images/orange.jpg")
    # TODO - 光波の複素振幅を設定する (2.29, 2.30, 2.31)
    u1_x, u1_y = np.meshgrid(x, y)  # u1座標

    # u2の作成
    u2_x, u2_y = np.meshgrid(x, y)  # u2座標

    # ① u1のゼロパディング
    u1_pad = np.pad(u1, W // 2)

    # ② ①をフーリエ変換する
    fa = np.fft.fft2(u1_pad)

    # ③ 伝達関数Ηを計算する
    dx = u2_x - u1_x * pp
    dy = u2_y - u1_y * pp
    r = np.sqrt(dx**2 + dy**2 + d**2)
    fb = h(z=d, λ=λ, r=r, W=W * 2, H=H * 2, pp=pp)

    # ④ ②と③を複素乗算する / ⑤ ④の結果を逆フーリエ変換する
    out = np.fft.ifft2((fa * fb))

    show_twin(np.angle(out), out)
    # np.angle(out) # 位相を見る時


if __name__ == "__main__":
    main()
