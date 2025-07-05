import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Load and prepare dataset
df = pd.read_csv("your_dataset.csv")
df = df[["Jitter", "Target"]].dropna()
X = df[["Jitter"]].to_numpy()
y = df["Target"].to_numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
os.makedirs("model", exist_ok=True)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Compute class weights
class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

# Model definition
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def clone_model(model):
    return copy.deepcopy(model)

# Stratified task sampler
def get_stratified_task(X, y, k_shot, q_query):
    classes = np.unique(y)
    X_task, y_task = [], []
    for cls in classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        n_needed = k_shot + q_query
        if len(X_cls) < n_needed:
            X_resampled, y_resampled = resample(X_cls, y_cls, replace=True, n_samples=n_needed, random_state=42)
        else:
            idx = np.random.choice(len(X_cls), n_needed, replace=False)
            X_resampled = X_cls[idx]
            y_resampled = y_cls[idx]
        X_task.append(X_resampled)
        y_task.append(y_resampled)

    return np.vstack(X_task), np.hstack(y_task)

# MAML training
def maml_train(model, X, y, inner_lr=0.01, outer_lr=0.001,
               epochs=100, n_tasks=20, k_shot=10, q_query=10):
    optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    meta_losses = []

    model.train()
    for epoch in range(epochs):
        meta_loss = 0.0
        for _ in range(n_tasks):
            X_task, y_task = get_stratified_task(X, y, k_shot, q_query)

            X_task_tensor = torch.tensor(X_task, dtype=torch.float32)
            y_task_tensor = torch.tensor(y_task, dtype=torch.long)

            n_classes = len(np.unique(y))
            k_total = k_shot * n_classes
            q_total = q_query * n_classes

            support_x = X_task_tensor[:k_total]
            support_y = y_task_tensor[:k_total]
            query_x = X_task_tensor[k_total:]
            query_y = y_task_tensor[k_total:]

            fast_model = clone_model(model)
            fast_optimizer = optim.SGD(fast_model.parameters(), lr=inner_lr)

            fast_model.train()
            fast_optimizer.zero_grad()
            support_preds = fast_model(support_x)
            support_loss = loss_fn(support_preds, support_y)
            support_loss.backward()
            fast_optimizer.step()

            fast_model.eval()
            query_preds = fast_model(query_x)
            query_loss = loss_fn(query_preds, query_y)
            meta_loss += query_loss

        meta_loss /= n_tasks
        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        meta_losses.append(meta_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Meta Loss: {meta_loss.item():.4f}")

    # Plot loss
    plt.plot(meta_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Meta Loss")
    plt.title("Meta Loss Over Time")
    plt.savefig("model/meta_loss_plot.png")
    plt.close()

# Train model
input_dim = X_train_scaled.shape[1]
output_dim = len(np.unique(y_train))
model = MLP(input_dim=input_dim, hidden_dim=128, output_dim=output_dim)

maml_train(model, X_train_scaled, y_train, inner_lr=0.05, outer_lr=0.005,
           epochs=100, n_tasks=16, k_shot=5, q_query=15)

# Save model
torch.save(model.state_dict(), "model/network_traffic_model.pth")

# Evaluation
def evaluate_model(model, X_data, y_true, name="Test"):
    model.eval()
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="weighted", zero_division=0)
    recall = recall_score(y_true, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, preds)
    cr = classification_report(y_true, preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, probs, multi_class="ovr")
    except:
        roc_auc = None

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"model/confusion_matrix_{name.lower()}.png")
    plt.close()

    return {
        "name": name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "report": cr,
        "conf_matrix": cm
    }

# Run evaluation
train_metrics = evaluate_model(model, X_train_scaled, y_train, name="Train")
test_metrics = evaluate_model(model, X_test_scaled, y_test, name="Test")

# Save metrics
with open("model/metrics_report.txt", "w") as f:
    for m in [train_metrics, test_metrics]:
        f.write(f"\n===== {m['name']} Metrics =====\n")
        f.write(f"Accuracy: {m['accuracy']:.4f}\n")
        f.write(f"Precision: {m['precision']:.4f}\n")
        f.write(f"Recall: {m['recall']:.4f}\n")
        f.write(f"F1 Score: {m['f1']:.4f}\n")
        if m["roc_auc"] is not None:
            f.write(f"ROC AUC: {m['roc_auc']:.4f}\n")
        else:
            f.write("ROC AUC could not be calculated.\n")
        f.write("Classification Report:\n" + m["report"] + "\n")

# Plot comparison
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
train_vals = [train_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]
test_vals = [test_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, train_vals, width, label="Train", color="skyblue")
plt.bar(x + width / 2, test_vals, width, label="Test", color="salmon")
plt.ylabel("Score")
plt.title("Train vs Test Evaluation Metrics")
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("model/metrics_comparison.png")
plt.close()
