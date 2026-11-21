import numpy as np
from training_loop import train_and_evaluate

seeds = [0, 1, 2, 3, 4]
accuracies = []

for seed in seeds:
    acc, correct, total = train_and_evaluate(seed)
    accuracies.append(acc)
    print(f"Seed {seed}: {acc:.2f}% ({correct}/{total})")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print("-" * 40)
print(f"Mean Validation Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")