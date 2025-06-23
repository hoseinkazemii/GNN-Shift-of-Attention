import os, math, random, torch, logging
from collections import defaultdict
import torch.nn as nn, torch.optim as optim
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs("./logs",  exist_ok=True)
os.makedirs("./plots", exist_ok=True)
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file    = f"./logs/rnn_training_log_{timestamp}.log"
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filemode="w")
console = logging.StreamHandler(); console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)


# SEQ_PATH = "./Data/29_participants/rnn_dataset.pt"          # (seqs, labels, source_files)
SEQ_PATH = "rnn_dataset.pt"          # (seqs, labels, source_files)

# seqs, labels, objects, srcs = torch.load(SEQ_PATH, weights_only=False)
seqs, labels, srcs = torch.load(SEQ_PATH, weights_only=False)
logging.info(f"Loaded {len(seqs):,} sequences from {SEQ_PATH}")

p2idx = defaultdict(list)
for i, s in enumerate(srcs):
    p2idx[s].append(i)

participants = sorted(p2idx.keys())
random.seed(42); random.shuffle(participants)
n_test  = math.ceil(len(participants) * 0.20)
test_p  = participants[:n_test]
train_p = participants[n_test:]
tr_idx  = [i for p in train_p for i in p2idx[p]]
te_idx  = [i for p in  test_p for i in p2idx[p]]

logging.info(f"Splitting {len(participants)} participants → "
             f"{len(train_p)} train, {len(test_p)} test")

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, labels, indices):
        self.x = [seqs[i]   for i in indices]
        self.y = [labels[i] for i in indices]
    def __len__(self):           return len(self.x)
    def __getitem__(self, idx):  return self.x[idx], torch.tensor(self.y[idx],
                                                                  dtype=torch.float)

train_ds, test_ds = SeqDataset(seqs, labels, tr_idx), SeqDataset(seqs, labels, te_idx)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test_ds , batch_size=32)

logging.info(f"Train samples: {len(train_ds):,}  "
             f"Test samples: {len(test_ds):,}")


y_train = torch.tensor([y for _, y in train_ds])
pos_w   = (y_train==0).sum() / (y_train==1).sum()
logging.info(f"pos_weight: {pos_w:.4f}")

INPUT_DIM  = train_ds[0][0].shape[-1]      # N_obj*10
HIDDEN_DIM = 158                           # param-matched

class FlatGRU(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, hid=HIDDEN_DIM, layers=1, p_dropout=0.30):
        super().__init__()
        self.gru   = nn.GRU(in_dim, hid, num_layers=layers,
                            batch_first=True, dropout=0. if layers==1 else p_dropout)
        self.drop  = nn.Dropout(p_dropout)
        self.head  = nn.Linear(hid, 1)
    def forward(self, x):                  # x:[B,T,in_dim]
        _, h = self.gru(x)                 # h:[layers,B,hid]
        h = self.drop(h[-1])
        return self.head(h).squeeze(1)     # logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = FlatGRU().to(device)
logging.info(f"\n{model}\nTotal parameters: "
             f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer  = optim.Adam(model.parameters(), lr=1e-3)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                  factor=0.5, patience=3, verbose=True)
criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss, preds, probs, labs = 0., [], [], []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        if train: optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward(); optimizer.step()
        tot_loss += loss.item()*x.size(0)
        p  = (out > 0).int().cpu()
        pr = torch.sigmoid(out).detach().cpu()
        preds.extend(p.tolist())
        probs.extend(pr.tolist())
        labs.extend(y.cpu().tolist())
    avg = tot_loss/len(loader.dataset)
    acc = accuracy_score(labs, preds)
    f1  = f1_score(labs, preds)
    auc = roc_auc_score(labs, probs)
    return avg, acc, f1, auc

EPOCHS = 30
history = {"tr":[], "te":[]}
logging.info("Starting training …")
for ep in range(1, EPOCHS+1):
    tr = run_epoch(train_dl, True)
    te = run_epoch(test_dl , False)
    scheduler.step(te[0])
    history["tr"].append(tr); history["te"].append(te)

    logging.info(
        f"Epoch {ep:02d} | "
        f"Train Loss {tr[0]:.4f} Acc {tr[1]:.3f} F1 {tr[2]:.3f} | "
        f"Test  Loss {te[0]:.4f} Acc {te[1]:.3f} F1 {te[2]:.3f} AUC {te[3]:.3f}"
    )


model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        all_preds.extend((torch.sigmoid(out) > 0.5).int().cpu().tolist())
        all_labels.extend(y.cpu().tolist())


cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Powerline","Powerline"])
plt.figure(figsize=(6,5)); disp.plot(cmap="Blues",values_format="d")
plt.title("RNN Confusion Matrix (Test)"); plt.tight_layout()
plt.savefig("./plots/rnn_confusion_matrix_test.png", dpi=300)
logging.info(f"Confusion matrix:\n{cm}")
logging.info("\nClassification Report (RNN, Test):\n"+
             classification_report(all_labels, all_preds,
                                   target_names=["No Powerline","Powerline"], digits=4))

def plot_metric(idx, label, ylabel):
    tr = [h[idx] for h in history["tr"]]
    te = [h[idx] for h in history["te"]]
    plt.figure(figsize=(7,5))
    plt.plot(range(1,EPOCHS+1), tr, label="Train")
    plt.plot(range(1,EPOCHS+1), te, label="Test")
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(label)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"./plots/rnn_{label.lower().replace(' ','_')}.png", dpi=300)
    plt.close()

plot_metric(0,"Loss","BCE Loss")
plot_metric(1,"Accuracy","Accuracy")
plot_metric(2,"F1 Score","F1 Score")
plot_metric(3,"AUC","AUC")

logging.info("Finished — all plots saved in ./plots and log in " + log_file)
