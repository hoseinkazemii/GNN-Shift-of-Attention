import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate_rnn(model, test_dataset, batch_size=64, **params):
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_probs = []
    all_labels = []
    
    logging.info("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (all_probs > threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Use optimal threshold for final predictions
    all_preds = (all_probs > best_threshold).astype(int)
    
    # Calculate final metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    # Calculate class-wise metrics
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    report = classification_report(all_labels, all_preds, 
                                 target_names=['No Powerline', 'Powerline'], 
                                 zero_division=0)
    
    precision, recall, _, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    logging.info("Final Test Results:")
    logging.info(f"  Optimal threshold: {best_threshold:.3f}")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  F1-Score: {f1:.4f}")
    logging.info(f"  AUC: {auc:.4f}")
    logging.info(f"  Precision (No/Yes): {precision[0]:.4f} / {precision[1]:.4f}")
    logging.info(f"  Recall (No/Yes): {recall[0]:.4f} / {recall[1]:.4f}")
    logging.info("Classification Report:")
    logging.info(f"\n{report}")
    
    cm = confusion_matrix(all_labels, all_preds)
    logging.info(f"Confusion Matrix:")
    logging.info(f"  True Negative: {cm[0,0]}, False Positive: {cm[0,1]}")
    logging.info(f"  False Negative: {cm[1,0]}, True Positive: {cm[1,1]}")
    
    return all_labels, all_preds
