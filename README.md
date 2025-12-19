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

### Get Started

```
# 点群法のサンプルコードを実行する
$ uv run python ./sample/pointcloud.py

# ホログラムを再生するサンプルコードを実行する
$ uv run python ./sample/reconst_hologram.py
```

<img width="800" height="94" alt="スクリーンショット 2025-11-28 4 09 12" src="https://github.com/user-attachments/assets/9eade545-98a2-45c7-81f2-09c1fa8d07dd" />


### Output Sample
- 生成されるホログラムの画像サンプル

#### 1点（ゾーンプレート）
<img width="450" alt="20251202_pointcloud" src="https://github.com/user-attachments/assets/01f36ba9-23d7-48fa-8cd9-d28cc01dcbce" />

#### 4点（ホログラム）
<img width="450" alt="スクリーンショット 2025-12-19 015815" src="https://github.com/user-attachments/assets/44f0064e-482d-4c96-9a46-83ce64bcc1b7" />

#### 四角形（ホログラム）
<img width="450" alt="rectholo" src="https://github.com/user-attachments/assets/559a47c5-7cca-47b0-a5e2-9f456f8df0de" />

### 四角形のホログラムと再生像
<img width="600" alt="Figure_1" src="https://github.com/user-attachments/assets/968afc69-3e8c-4cf4-8fb4-da25793ceb4b" />
<EOD>









