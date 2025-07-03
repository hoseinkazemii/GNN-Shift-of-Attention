import torch
from sklearn.metrics import classification_report
import logging


def evaluate_rnn(model, test_dataset, **params):
    device = params.get("device")
    batch_size = params.get("batch_size")

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = (torch.sigmoid(out) > 0.5).int().cpu().tolist()
            labels = y.cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

    report = classification_report(all_labels, all_preds, target_names=["No Powerline", "Powerline"], digits=4)
    logging.info("\nClassification Report (RNN, Test):\n" + report)

    return all_labels, all_preds
