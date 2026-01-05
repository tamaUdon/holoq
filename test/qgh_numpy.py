def bit_k(p, k):
    # T(•)の実装
    # k番目のbit値を見る
    return (p >> k) & 1  # k個右シフトして右からk桁目を取り出す


def qgh_pixel_single_point(xj, xh, yj, yh, rho_j, k):
    # Quantum Computer-Generated Holography の式 (2) の実装 (Σ以外)
    dx = xh - xj
    dy = yh - yj
    r_2 = dx * dx + dy * dy
    p = rho_j * r_2
    return bit_k(p, k)


def qgh_image_loop():
    ...
    # XJ, YJ, xh, yhを配列にする
    # loopでqgh_pixel_single_pointを回して同じbitを見た場合のpを作る
    # 足し合わせて最終的なpを作る


# 1点
xj, yj = (1, 2)
xh, yh = (3, 1)
rho_j = 2
k = 3  # 1,2,3

# この場合、見る桁kを0,1,2,3と変えると0,1,0,1と出力される
# 10 -> 1010(2bit) の右から1つずつ取り出す操作
print(qgh_pixel_single_point(xh, yh, xj, yj, rho_j, k))
