metrics_text = (
    f"Accuracy: {acc:.2f}\n"
    f"Balanced Acc: {bal_acc:.2f}\n"
    f"F1 Score: {f1:.2f}\n"
    f"Precision: {precision:.2f}\n"
    f"Recall: {recall:.2f}"
)

fig.text(0.97, 0.5, metrics_text, fontsize=10, va='center', ha='left')

plt.tight_layout(rect=[0, 0, 0.93, 1])