from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List
import time

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / 'data' / 'input'
OUTPUT_DIR = ROOT_DIR / 'data' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 学習データ
x_train = np.load(DATA_DIR / 'x_train.npy', allow_pickle=True)
t_train = np.load(DATA_DIR / 't_train.npy', allow_pickle=True)

# 検証データを取る
x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.2, random_state=seed)

# テストデータ
x_test = np.load(DATA_DIR / 'x_test.npy', allow_pickle=True)


def text_transform(text: List[int], max_length=512):
    # <BOS>はすでに1で入っている．<EOS>は2とする．
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

    # NOTE: 宿題用データセットでは<PAD>は3です．
    return torch.tensor(label_list), pad_sequence(text_list, padding_value=3).T, torch.tensor(len_seq_list)


word_num = np.concatenate(np.concatenate((x_train, x_test))).max() + 1

"""### 実装"""

batch_size = 128

train_dataloader = DataLoader(
    [(t, x) for t, x in zip(t_train, x_train)],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch,
)
valid_dataloader = DataLoader(
    [(t, x) for t, x in zip(t_valid, x_valid)],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)
test_dataloader = DataLoader(
    x_test,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, padding_idx: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class Attention(nn.Module):
    def __init__(self, hid_dim: int):
        super().__init__()
        self.attn = nn.Linear(hid_dim, 1)

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attn(lstm_output).squeeze(2)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e10)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context


class SequenceTaggingNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embedding = Embedding(vocab_size, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = Attention(hid_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, 1)

    def forward(self, x: torch.Tensor, _: torch.Tensor = None, lengths: torch.Tensor = None) -> torch.Tensor:
        emb = self.embedding(x)
        emb = self.emb_dropout(emb)

        if lengths is not None:
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_outputs, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            output, _ = self.lstm(emb)

        seq_len = output.size(1)
        mask = torch.arange(seq_len, device=x.device)[None, :] < lengths[:, None]
        attn_output = self.attention(output, mask)
        features = self.dropout(attn_output)
        logits = self.fc(features)
        return logits.squeeze(1)

def log(message: str):
    print(message, flush=True)


emb_dim = 100
hid_dim = 128
n_epochs = 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log(f"単語種数: {word_num}")
log(f"Using device: {device}")
log(f"Train size: {len(x_train)}, Valid size: {len(x_valid)}, Test size: {len(x_test)}")
log(f"Hyperparameters -> emb_dim: {emb_dim}, hid_dim: {hid_dim}, batch_size: {batch_size}, lr: 1e-3, epochs: {n_epochs}")
log(f"Hyperparameters -> emb_dim: {emb_dim}, hid_dim: {hid_dim}, batch_size: {batch_size}, lr: 1e-3, epochs: {n_epochs}")

net = SequenceTaggingNet(word_num, emb_dim, hid_dim, num_layers=2, dropout=0.5)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
criterion = nn.BCEWithLogitsLoss()

best_valid_f1 = -1.0
best_model_path = OUTPUT_DIR / 'best_model.pth'
training_log_path = OUTPUT_DIR / 'training_log.csv'
if training_log_path.exists():
    training_log_path.unlink()
log_columns = [
    "epoch",
    "train_loss",
    "train_acc",
    "train_f1",
    "valid_loss",
    "valid_f1",
    "best_valid_f1",
    "epoch_time_sec",
    "learning_rate",
]
pd.DataFrame(columns=log_columns).to_csv(training_log_path, index=False)
epoch_history = []

for epoch in range(n_epochs):
    epoch_start = time.time()
    losses_train = []
    losses_valid = []

    net.train()
    n_train = 0
    acc_train = 0
    t_train_epoch = []
    y_train_epoch = []
    for label, line, len_seq in train_dataloader:

        t = label.to(device).float()
        x = line.to(device)
        len_seq = len_seq.to(device)

        optimizer.zero_grad()
        logits = net(x, None, len_seq)
        loss = criterion(logits, t)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits).round()
            acc_train += (preds == t).sum().item()
            t_train_epoch.extend(t.cpu().tolist())
            y_train_epoch.extend(preds.cpu().tolist())

        losses_train.append(loss.item())

        n_train += t.size(0)

    # Valid
    t_valid = []
    y_pred = []
    net.eval()
    for label, line, len_seq in valid_dataloader:

        t = label.to(device).float()
        x = line.to(device)
        len_seq = len_seq.to(device)

        with torch.no_grad():
            logits = net(x, None, len_seq)
            loss = criterion(logits, t)
            y = torch.sigmoid(logits)
            pred = y.round()

        t_valid.extend(t.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

        losses_valid.append(loss.item())

    train_acc = acc_train / n_train if n_train else 0.0
    train_f1 = f1_score(t_train_epoch, y_train_epoch, average='macro') if t_train_epoch else 0.0
    valid_f1 = f1_score(t_valid, y_pred, average='macro') if t_valid else 0.0
    epoch_time = time.time() - epoch_start
    avg_train_loss = float(np.mean(losses_train)) if losses_train else 0.0
    avg_valid_loss = float(np.mean(losses_valid)) if losses_valid else 0.0
    current_lr = scheduler.get_last_lr()[0]

    log(f"[Epoch {epoch + 1:02d}/{n_epochs}] {epoch_time:.1f}s, lr={current_lr:.6f}")
    log(f"  Train -> Loss: {avg_train_loss:.3f}, Acc: {train_acc:.3f}, F1: {train_f1:.3f}")
    log(f"  Valid -> Loss: {avg_valid_loss:.3f}, F1: {valid_f1:.3f}, Best F1: {max(best_valid_f1, valid_f1):.3f}")

    epoch_record = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
        "valid_loss": avg_valid_loss,
        "valid_f1": valid_f1,
        "best_valid_f1": best_valid_f1,
        "epoch_time_sec": epoch_time,
        "learning_rate": current_lr,
    }
    epoch_history.append(epoch_record)
    pd.DataFrame([epoch_record]).to_csv(
        training_log_path, mode="a", header=False, index=False
    )

    scheduler.step()

    if valid_f1 > best_valid_f1:
        best_valid_f1 = valid_f1
        torch.save(net.state_dict(), best_model_path)
        log(f"    -> Best Model Saved! (F1: {valid_f1:.3f})")
    else:
        log(f"    -> No improvement for this epoch (current best: {best_valid_f1:.3f})")

log(f"Loading best model with F1: {best_valid_f1:.3f}")
net.load_state_dict(torch.load(best_model_path))

net.eval()

y_pred = []
for _, line, len_seq in test_dataloader:

    x = line.to(device)
    len_seq = len_seq.to(device)

    with torch.no_grad():
        h = net(x, None, len_seq)
        y = torch.sigmoid(h)

    pred = y.round().squeeze()  # 0.5以上の値を持つ要素を正ラベルと予測する

    y_pred.extend(pred.tolist())


submission = pd.Series(y_pred, name='label')
submission_path = OUTPUT_DIR / 'submission_pred_07.csv'
submission.to_csv(submission_path, header=True, index_label='id')

