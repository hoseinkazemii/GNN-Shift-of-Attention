import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import networkx as nx


def plot_graph(G):
    plt.figure(figsize=(18, 12))
    pos = nx.spring_layout(G, seed=42, k=0.5)  # `k` controls spacing between nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightgreen", node_size=700, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6)
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_family="sans-serif",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.8)
    )
    plt.title("Static Graph — Fixated Objects (≥0.1s), Loading Started, Distance < 50", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./plots/static_graph.png", dpi=600)
    plt.show()


def plot_metrics(history, **params):
    model_name = params.get("model_name")

    metrics = ["loss", "acc", "f1", "auc"]
    epochs = range(1, len(history["train_loss"]) + 1)

    for metric in metrics:
        key = metric.lower().replace(" ", "_")
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history[f"train_{key}"], label=f"Train {metric}")
        plt.plot(epochs, history[f"test_{key}"], label=f"Test {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./plots/{key}_per_epoch_{model_name}.png", dpi=300)
        plt.close()


def plot_confusion(true_labels, predicted_labels, **params):
    model_name = params.get("model_name")
    save_path = f"./plots/{model_name}_confusion_matrix_test_set.png"

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Powerline", "Powerline"])

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix on Test Set")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
