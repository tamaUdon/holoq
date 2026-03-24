# holoq ドキュメント

## 1. 概要

このリポジトリは、計算機生成ホログラム (CGH: Computer-Generated Holography) と、
量子回路を用いた量子ホログラム生成 (QGH: Quantum-Generated Holography) の試作・検証を行うための研究用コードです。

大きく次の2つを含みます。

- 古典計算による CGH の生成と再生
- Qiskit を用いた QGH 回路の構築とシミュレーション

実装の中心は `sample/`
`sample/pointcloud.py` ... 古典CGH
`sample/qgh.py` ...QGH
`test/` ...量子画像処理の検証用コード、試行錯誤の途中経過のログ

## 2. 何をしているか

### 2.1 古典 CGH

古典 CGH 側では、点群データからホログラム面上の干渉パターンを数値計算で生成します。主な流れは以下です。

1. 物体点群を用意する
2. ホログラム面上の各画素について、物体点からの距離に応じた位相項を計算する
3. 各点の寄与を重ね合わせてホログラムを作る
4. 必要に応じてフレネル回折計算により再生像を確認する

`sample/pointcloud.py` ...基本的な点群法によるホログラム生成
`sample/reconst_hologram.py` ...再生像確認を
`sample/monopolar.py` ...高速化手法の検証を行っています。

### 2.2 量子 CGH

QGHのコードでは、点群情報とホログラム面座標を量子レジスタに載せ、
加算器・乗算器を組み合わせた量子回路として1画素分の計算を組み立てています。
主な流れは以下です。

1. 点群座標と位相パラメータを量子状態として初期化する
2. ホログラム面座標との差分を加算器で求める
3. 差分の二乗和を作る
4. 位相係数を掛ける
5. 測定結果から画素値を取り出す

`sample/qgh.py` ...Qiskit の `DraperQFTAdder` と `RGQFTMultiplier` を用いて、QGH の 1 画素計算を画素ごとに回路実行する形で試作しています。

## 3. ディレクトリ・ファイル説明

### 3.1 ルート

- `README.md`
  - セットアップ手順、サンプル実行方法、出力例を簡単にまとめたファイルです。

- `pyproject.toml`
  - Python バージョン要件と依存ライブラリ定義です。uvで管理しています。

- `uv.lock`
  - `uv` 用のロックファイルです。依存関係の再現に使います。

- `main.py`
  - 現状は最小のエントリポイントのみで、研究実装の本体ではありません。

- `docs.md`
  - 本ドキュメントです。

### 3.2 `sample/`

- `sample/constants.py`
  - 定数定義です。
  - `ClassicalConstants` ...古典CGH用の定数定義
  - `QuantumConstants` ...量子回路のビット幅やホログラムサイズを計算するための定数定義

- `sample/pointcloud.py`
  - 古典CGHの中心的なサンプルです。
  - 単一点、4点、矩形輪郭などの点群を生成し、点群法でホログラムを計算します。
  - 現状の古典側の主実装として見るのが妥当です。

- `sample/reconst_hologram.py`
  - 生成したホログラムからフレネル FFT により再生像を確認するためのスクリプトです。
  - ただし現状は `Constants` 参照など、コード更新に追従していない部分があり、そのままでは修正なしに動かない可能性があります。

- `sample/monopolar.py`
  - monopolar/bipolar 近似を意識した高速化検証用コードです。
  - 古典的な位相計算をビット演算寄りに近似する実験的実装です。
  - こちらも `Constants` 参照など、現状の定数定義と整合していない可能性があります。

- `sample/bunny.py`
  - Open3Dのbunnyデータセットを点群として読み込み、ダウンサンプリング後にホログラム生成へ流すサンプルです。
  - 外部点群データを入力した場合の動作検証用です。

- `sample/qgh.py`
  - QGH の主実装です。
  - 量子レジスタ定義、初期状態準備、加算器・乗算器の接続、シミュレータ実行、測定結果のログ出力、ホログラム配列生成までを含みます。
  - 現時点では小規模な画素数・物体点数での検証を前提にした実装です。

### 3.3 `test/`

- `test/qgh.py`
  - 初期の量子ホログラム試作コードです。
  - 独自の量子状態表現や GHZ 状態生成を試しており、現在の主実装というよりは検討履歴に近い位置づけです。

- `test/qgh_numpy.py`
  - QGH の数式の一部を NumPy ベースで単純化して確かめる小さな検証コードです。
  - 1 画素計算の式確認用です。

- `test/qgh_gpt.py`
  - 点群情報を量子レジスタへ basis encoding する方向の検証コードです。
  - QGH 本体というより、入力表現の設計確認用です。

- `test/qtest.py`
  - Qiskit の基本操作、状態ベクトル、測定、ユニタリ演算の挙動確認用スクリプトです。

- `test/NEQRtest.ipynb`
  - NEQRなど量子画像表現に関するノートブック形式の検証ファイルです。

### 3.4 `quantum-image-processing/`

- `quantum-image-processing/README.md`
- `quantum-image-processing/main.py`
- `quantum-image-processing/encoder.py`
- `quantum-image-processing/circuit.py`
- `quantum-image-processing/test.png`
- `quantum-image-processing/LICENSE`

上記は量子画像処理に関する参照用コード群です。
QGH実装の補助的な参考資料として置いてあり、このリポジトリ固有の主成果ではありません。

## 4. 実行環境・前提条件

- Python 3.11
- `uv` 推奨
- QGHは `qiskit-aer` によるシミュレーションを想定

## 5. 注意事項

- 実機実行ではなくシミュレータ中心のため、実機ノイズやハードウェア制約は十分には反映していません。
  <EOD>
