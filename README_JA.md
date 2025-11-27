**日本語** | [English](README.md)

# 東京大学 Deep Learning Course Competition

## コンペ結果

- **最終順位**: **2位** / 1,263 名
- **LB スコア**: **0.93082**

## 概要

Fashion-MNIST（10クラス）のファッション版 MNIST を、マルチレイヤー・パーセプトロンで分類しました。

Fashion-MNIST の詳細はこちら:  
Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist

## ルール

- すべてのスクリプトを 1 つのファイルにまとめてください。
- `x_train` と `t_train` 以外の学習データは使用しないでください。

## アプローチ

- データ前処理
  - `data/input/` から `x_train.npy`, `t_train.npy`, `x_test.npy` を読み込む。
  - `train_test_split`（`random_state=42`）で学習データの 10% を検証用に分割。
  - `collate_batch` 内の `text_transform` で長文を両端カット → EOS(id=2) を付与 → id=3 でパディング。

- 語彙数とバッチ生成
  - 学習・テストの全トークンから語彙サイズを求め、Embedding に渡す。
  - `DataLoader` は学習のみ shuffle、検証/テストは順序固定。

- モデル（`src/lecture07_homework.py` の `SequenceTaggingNet`）
  - padding id=3 の Embedding → ガウスノイズ注入（std=0.2）→ Spatial Dropout。
  - 隠れ 256、ドロップアウト 0.3 の双方向 2 層 GRU（直交初期化）。
  - 双方向出力を LayerNorm → マスク付き平均プーリング & マスク付き最大プーリング。
  - プーリング結果（512 次元）を結合し、線形層でロジット出力。

- 学習設定
  - 各シード 18 epoch、AdamW（lr `1.5e-3`, weight_decay `1e-4`）+ `OneCycleLR`（pct_start 0.2, cosine）。
  - `BCEWithLogitsLoss` によるラベルスムージング（0.05）、勾配クリップ 1.0、各 epoch で macro F1 を記録。
  - 最良の検証 F1 チェックポイントを保存し、推論前に復元。

- アンサンブルと推論
  - シード `[42, 2025, 777, 1234]` で同じ学習ループを実行。
  - 各ベストモデルからテストローダのシグモイド確率を取得し平均化、0.5 で二値化。
  - 予測を `data/output/submission_seed_ensemble.csv` に保存（Kaggle 形式）。

## 使用技術

- Python 3.9+
- PyTorch（テンソル演算、GRU、Optimizer、Scheduler）
- NumPy（配列処理・アンサンブル計算）
- scikit-learn（`train_test_split`, `f1_score`）
- pandas（提出ファイル生成）

