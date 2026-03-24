### Prerequisite
- Python >=3.11
- uv (推奨)

### Install

```
# uv が入っていなければインストール
$ pip install uv

# 依存関係のインストール
$ uv sync
```

### Get Started (CGH)

```
# 点群法のサンプルコードを実行する
$ uv run python ./sample/pointcloud.py

# ホログラムを再生するサンプルコードを実行する
$ uv run python ./sample/reconst_hologram.py
```

<img width="800" height="94" alt="スクリーンショット 2025-11-28 4 09 12" src="https://github.com/user-attachments/assets/9eade545-98a2-45c7-81f2-09c1fa8d07dd" />

### Get Started (QGH)

```
# QGHを実行する
$ uv run python ./sample/qgh.py
```
<img width="800" height="122" alt="スクリーンショット 2026-03-24 20 48 43" src="https://github.com/user-attachments/assets/07143a5f-8800-4562-96e3-43a89c3f1d5c" />


### Output Sample (QGH)
### 出力 N=4, X=4の場合
- 4*4画素で出力したもの
- 測定のたびに出力は変わる
<img width="400" alt="holoq_n4_x4" src="https://github.com/user-attachments/assets/0b6f02f7-d462-4cc2-a8cc-2b6592850eee" />

#### 点群数2, 6shotsの場合のヒストグラム
- QGHの測定後、0|1の確率をプロットしたもの
<img width="400" height="480" alt="measured" src="https://github.com/user-attachments/assets/9257e576-1d19-4a50-a979-9f0623098be0" />

### Output Sample (CGH)
- 生成されるホログラムの画像サンプル

#### 1点（ゾーンプレート）
<img width="400" alt="20251202_pointcloud" src="https://github.com/user-attachments/assets/01f36ba9-23d7-48fa-8cd9-d28cc01dcbce" />

#### 4点（ホログラム）
<img width="400" alt="スクリーンショット 2025-12-19 015815" src="https://github.com/user-attachments/assets/44f0064e-482d-4c96-9a46-83ce64bcc1b7" />

#### 四角形のホログラムと再生像
<img width="600" alt="スクリーンショット 2025-12-22 015158" src="https://github.com/user-attachments/assets/c4ce2166-94d4-48e7-a96d-286bbccaddd7" />

#### Monopolar高速化手法によるホログラム
<img width="400" alt="monopolar_hologram" src="https://github.com/user-attachments/assets/14f7b3aa-e23c-4c64-84ea-da0d2fe6db9a" />
<EOD>















