import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [
    "VGG16", "VGG19", "Inception V3", 
    "HOG-RGB", "Resnet50", "Shallow CNN"
]

# Accuracy (mean ± std)
accuracy_means = [0.42, 0.77, 0.77, 0.64, 0.52, 0.74]
accuracy_stds  = [0.004, 0.039, 0.024, 0.021, 0.188, 0.088]

# F1-score (mean ± std)
f1_means = [0.34, 0.76, 0.75, 0.52, 0.49, 0.73]
f1_stds  = [0.016, 0.051, 0.042, 0.071, 0.192, 0.097]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracy_means, width, yerr=accuracy_stds, capsize=5, label='Accuracy', color='#90CAF9')
bars2 = ax.bar(x + width/2, f1_means, width, yerr=f1_stds, capsize=5, label='F1 Score', color='#A5D6A7')

# Labels and formatting
ax.set_ylabel('Score')
ax.set_title('Model Selection Results (Accuracy and F1 Score with Std Dev)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30)
ax.set_ylim(0, 1.1)
ax.legend()

plt.tight_layout()
plt.savefig("hardcoded_model_selection_results_barplot.png")
plt.show()
