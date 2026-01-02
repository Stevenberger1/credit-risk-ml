import matplotlib.pyplot as plt
import seaborn as sns


def plot_roc_curves(all_results):
    if not all_results:
        raise ValueError("No results provided for ROC plotting.")

    plt.figure(figsize=(8, 6))

    for res in all_results:
        fpr = res["roc_curve"]["fpr"]
        tpr = res["roc_curve"]["tpr"]
        auc = res["metrics"]["roc_auc"]

        plt.plot(fpr, tpr, label=f"{res['model_name']} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pr_curves(all_results):
    if not all_results:
        raise ValueError("No results provided for PR plotting.")

    plt.figure(figsize=(8, 6))

    for res in all_results:
        precision = res["pr_curve"]["precision"]
        recall = res["pr_curve"]["recall"]
        pr_auc = res["metrics"]["pr_auc"]

        plt.plot(recall, precision, label=f"{res['model_name']} (PR AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(all_results):
    if not all_results:
        raise ValueError("No results provided for confusion matrix plotting.")

    for res in all_results:
        cm = res["confusion_matrix"]

        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

        plt.title(f"Confusion Matrix\n{res['model_name']}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()


def display_summary_table(results_df):
    try:
        from IPython.display import display
        display(results_df)
    except ImportError:
        print(results_df)
