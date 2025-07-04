import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
from tqdm import tqdm

class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance better than weighted BCE
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def create_balanced_sampler(dataset):
    labels = dataset.y_tensor.long().tolist()  # Fast and clean
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def train_rnn(model, train_dataset, val_dataset, test_dataset, pos_weight=None,
              batch_size=64, num_epochs=30, learning_rate=0.0005, 
              weight_decay=1e-3, early_stopping_patience=8, 
              use_focal_loss=False, use_balanced_sampling=True, **params):
    
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loaders
    if use_balanced_sampling:
        sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, verbose=True, min_lr=1e-6)

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [],
        'test_loss': [], 'test_acc': [], 'test_f1': [], 'test_auc': []
    }

    best_val_f1 = 0.0
    patience_counter = 0

    logging.info(f"Starting training for {num_epochs} epochs")
    logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_preds, train_labels, train_probs = [], [], [], []

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())
                train_probs.extend(probs.cpu().numpy())

        # Compute training metrics
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        try:
            train_auc = roc_auc_score(train_labels, train_probs)
        except ValueError:
            train_auc = 0.0

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)

        # --- Validation ---
        model.eval()
        val_losses, val_preds, val_labels, val_probs = [], [], [], []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.0

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        scheduler.step(val_f1)

        acc_gap = train_acc - val_acc
        f1_gap = train_f1 - val_f1

        logging.info(f"Epoch {epoch+1}:")
        logging.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        logging.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        logging.info(f"  Gaps  - Acc: {acc_gap:.4f}, F1: {f1_gap:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model_name = params.get("model_name", "improved_rnn_model")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_f1': best_val_f1,
                'history': history
            }, f"./models/{model_name}_best.pth")
            logging.info(f"  New best model saved! (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1

        if acc_gap > 0.3 or f1_gap > 0.3:
            logging.warning(f"  WARNING: Potential overfitting! (Acc gap: {acc_gap:.4f}, F1 gap: {f1_gap:.4f})")

        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Load best model
    model_name = params.get("model_name")
    checkpoint = torch.load(f"./models/{model_name}_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    model.eval()
    test_losses, test_preds, test_labels, test_probs = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            test_losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_loss = np.mean(test_losses)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    try:
        test_auc = roc_auc_score(test_labels, test_probs)
    except ValueError:
        test_auc = 0.0

    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    history['test_f1'].append(test_f1)
    history['test_auc'].append(test_auc)

    logging.info(f"\nFinal Test Evaluation:")
    logging.info(f"  Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

    return history
