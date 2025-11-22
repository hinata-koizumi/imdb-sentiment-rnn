from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List
import time
import os

# --- パス設定 ---
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / 'data' / 'input'
OUTPUT_DIR = ROOT_DIR / 'data' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- シード固定 ---
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- データ読み込み ---
x_train = np.load(DATA_DIR / 'x_train.npy', allow_pickle=True)
t_train = np.load(DATA_DIR / 't_train.npy', allow_pickle=True)
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.2, random_state=seed)
x_test = np.load(DATA_DIR / 'x_test.npy', allow_pickle=True)

def text_transform(text: List[int], max_length=512):
    text = text[:max_length - 1] + [2]
    return text, len(text)

def collate_batch(batch):
    label_list, text_list, len_seq_list = [], [], []
    for sample in batch:
        if isinstance(sample, tuple):
            label, text = sample
            label_list.append(label)
        else:
            text = sample.copy()
        text, len_seq = text_transform(text)
        text_list.append(torch.tensor(text))
        len_seq_list.append(len_seq)
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3).T, torch.tensor(len_seq_list)

word_num = np.concatenate(np.concatenate((x_train, x_test))).max() + 1

# ==========================================
# モデル定義: GRU + GaussianNoise + Corrected Pooling
# ==========================================

class GaussianNoise(nn.Module):
    """学習時のみEmbeddingにノイズを乗せて頑健性を高める"""
    def __init__(self, stddev=0.1):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, emb, seq]
        x = x.unsqueeze(3)      # [batch, emb, seq, 1]
        x = super(SpatialDropout, self).forward(x)
        x = x.squeeze(3)
        x = x.permute(0, 2, 1)
        return x

class SequenceTaggingNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=3)
        
        # 【追加】Embeddingへのノイズ注入
        self.gaussian_noise = GaussianNoise(stddev=0.2)
        
        self.spatial_dropout = SpatialDropout(dropout)
        
        # 【変更】LSTM -> GRU (パラメータ削減と収束安定化)
        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        
        self.layer_norm = nn.LayerNorm(hid_dim * 2)
        self.fc = nn.Linear(hid_dim * 4, 1) # Avg + Max

        self._init_weights()

    def _init_weights(self):
        # 直交初期化
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor, _: torch.Tensor = None, lengths: torch.Tensor = None) -> torch.Tensor:
        emb = self.embedding(x)
        emb = self.gaussian_noise(emb) # Noise
        emb = self.spatial_dropout(emb)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_outputs, _ = self.gru(packed)
            output, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            output, _ = self.gru(emb)
        
        # output: [batch, seq_len, hid_dim * 2]
        output = self.layer_norm(output)

        # --- 【修正】Masked Average Pooling ---
        # pad_packed_sequenceはパディング部分を0埋めして返す。
        # 単純にsumをとると有効部分の合計になる。
        sum_pool = torch.sum(output, dim=1) 
        
        # 分母を実際の長さにする (ブロードキャストのためunsqueeze)
        actual_lengths = lengths.unsqueeze(1).float().to(output.device)
        avg_pool = sum_pool / actual_lengths # パディングを除外した正しい平均
        
        # Max Pooling
        # パディング部分(0)が最大値にならないよう注意が必要だが、
        # LayerNorm後やReLU等の活性化によっては0より大きい値が出るため、
        # 本来はマスクして-infを入れるのが厳密。
        # ただしGRU生出力なら0付近が中心なので、簡易的にそのままMaxをとるか、
        # 以下のようにマスク処理を行う。
        
        # マスク作成 (seq_len次元)
        mask = torch.arange(output.size(1), device=output.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(2) # [batch, seq_len, 1]
        
        # マスク外を非常に小さい値にしてMaxに選ばれないようにする
        output_masked = output.masked_fill(~mask, -1e9)
        max_pool, _ = torch.max(output_masked, dim=1)

        # 結合
        concatenated = torch.cat((avg_pool, max_pool), 1)
        
        logits = self.fc(concatenated)
        return logits.squeeze(1)

def log(message: str):
    print(message, flush=True)

# ==========================================
# 設定 & 学習ループ
# ==========================================
emb_dim = 300
hid_dim = 256
batch_size = 128
n_epochs = 18  # ノイズが入るため少し長めに
max_lr = 1.5e-3 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")

net = SequenceTaggingNet(word_num, emb_dim, hid_dim, num_layers=2, dropout=0.3)
net.to(device)

optimizer = optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-4) # Weight Decay強め
criterion = nn.BCEWithLogitsLoss()

steps_per_epoch = len(x_train) // batch_size + 1
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    epochs=n_epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.2,
    anneal_strategy='cos'
)

best_valid_f1 = -1.0
best_model_path = OUTPUT_DIR / 'best_model_final.pth'

# DataLoader
train_dataloader = DataLoader([(t, x) for t, x in zip(t_train, x_train)], batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader([(t, x) for t, x in zip(t_valid, x_valid)], batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(x_test, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

def smooth_labels(labels, smoothing=0.05):
    return labels * (1 - smoothing) + 0.5 * smoothing

for epoch in range(n_epochs):
    epoch_start = time.time()
    losses_train = []
    
    net.train()
    t_train_epoch, y_train_epoch = [], []
    
    for label, line, len_seq in train_dataloader:
        t = label.to(device).float()
        x = line.to(device)
        len_seq = len_seq.to(device)
        
        t_smooth = smooth_labels(t, smoothing=0.05)

        optimizer.zero_grad()
        logits = net(x, None, len_seq)
        loss = criterion(logits, t_smooth)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits).round()
            t_train_epoch.extend(t.cpu().tolist())
            y_train_epoch.extend(preds.cpu().tolist())
        losses_train.append(loss.item())

    # Valid
    losses_valid = []
    t_valid_epoch, y_pred_valid = [], []
    net.eval()
    for label, line, len_seq in valid_dataloader:
        t = label.to(device).float()
        x = line.to(device)
        len_seq = len_seq.to(device)
        with torch.no_grad():
            logits = net(x, None, len_seq)
            loss = criterion(logits, t)
            pred = torch.sigmoid(logits).round()
        
        t_valid_epoch.extend(t.cpu().tolist())
        y_pred_valid.extend(pred.cpu().tolist())
        losses_valid.append(loss.item())

    train_f1 = f1_score(t_train_epoch, y_train_epoch, average='macro')
    valid_f1 = f1_score(t_valid_epoch, y_pred_valid, average='macro')
    epoch_time = time.time() - epoch_start
    avg_train_loss = float(np.mean(losses_train))
    avg_valid_loss = float(np.mean(losses_valid))
    
    log(f"[Epoch {epoch+1:02d}/{n_epochs}] {epoch_time:.1f}s")
    log(f"  Train -> Loss: {avg_train_loss:.3f}, F1: {train_f1:.3f}")
    log(f"  Valid -> Loss: {avg_valid_loss:.3f}, F1: {valid_f1:.3f}, Best: {max(best_valid_f1, valid_f1):.3f}")

    if valid_f1 > best_valid_f1:
        best_valid_f1 = valid_f1
        torch.save(net.state_dict(), best_model_path)
        log(f"    -> Saved Best Model! (F1: {valid_f1:.3f})")

# 推論
log(f"Loading best model: {best_valid_f1:.3f}")
if best_model_path.exists():
    net.load_state_dict(torch.load(best_model_path))
net.eval()

y_pred_test = []
for _, line, len_seq in test_dataloader:
    x = line.to(device)
    len_seq = len_seq.to(device)
    with torch.no_grad():
        logits = net(x, None, len_seq)
        y = torch.sigmoid(logits)
    pred = y.round().squeeze()
    y_pred_test.extend(pred.tolist())

submission = pd.Series(y_pred_test, name='label')
submission_path = OUTPUT_DIR / 'submission_pred_final_v2.csv'
submission.to_csv(submission_path, header=True, index_label='id')
log(f"Done. Submission saved.")

