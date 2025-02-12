import torch
import matplotlib.pyplot as plt

import numpy as np

metrics_normal = torch.load("C:/Users/Asus/Desktop/GTSRBSAM/history/training_metrics_normal.pth")
metrics_sam = torch.load("C:/Users/Asus/Desktop/GTSRBSAM/history/training_metrics_sam.pth")

num_epochs = 20
num_batches_per_epoch = len(metrics_normal["batch_loss"]) // num_epochs

# Aggregate batch metrics into epoch metrics
def aggregate_epoch_metrics(batch_metrics, num_batches_per_epoch):
    return [
        np.mean(batch_metrics[i * num_batches_per_epoch : (i + 1) * num_batches_per_epoch])
        for i in range(num_epochs)
    ]

normal_epoch_loss = aggregate_epoch_metrics(metrics_normal["batch_loss"], num_batches_per_epoch)
sam_epoch_loss = aggregate_epoch_metrics(metrics_sam["batch_loss"], num_batches_per_epoch)

normal_epoch_accuracy = aggregate_epoch_metrics(metrics_normal["batch_accuracy"], num_batches_per_epoch)
sam_epoch_accuracy = aggregate_epoch_metrics(metrics_sam["batch_accuracy"], num_batches_per_epoch)


plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), normal_epoch_loss, label="Normal Loss", color="blue", marker="o")
plt.plot(range(1, num_epochs + 1), sam_epoch_loss, label="SAM Loss", color="orange", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid()
plt.savefig("loss_per_epoch.png", dpi=300)  

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), normal_epoch_accuracy, label="Normal Accuracy", color="green", marker="o")
plt.plot(range(1, num_epochs + 1), sam_epoch_accuracy, label="SAM Accuracy", color="red", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy per Epoch")
plt.legend()
plt.grid()
plt.savefig("accuracy_per_epoch.png", dpi=300)  
plt.show()
