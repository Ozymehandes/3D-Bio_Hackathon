from neural_net import *
import numpy as np
import torch

def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray, thresh: float = 0.5) -> float:
    """
    Binary-classification accuracy given predicted probabilities and true labels.
    """
    preds = (probs >= thresh).astype(int)
    return (preds == labels).mean()

train = np.load("data/splits/train.npz", allow_pickle=True)
val   = np.load("data/splits/val.npz",   allow_pickle=True)
test  = np.load("data/splits/test.npz",  allow_pickle=True)

n_pos_train = (train["y"] == 1).sum()
n_neg_train = (train["y"] == 0).sum()

batch_size = 64
epochs = 60
lr = 1e-4
hidden_dim = 512
dropout = 0.4
# Prepare a Dataloader and create model
train_loader = prepare_train_loader(train["X"], train["y"], n_neg_train, n_pos_train, batch_size=batch_size)
network = SimpleDenseNet(esm_emb_dim=1280, hidden_dim=hidden_dim, dropout=dropout)
trained_network = train_net(network, train_loader, num_epochs=epochs, lr=lr)

from sklearn.metrics import precision_recall_curve, f1_score
val_probs = torch.sigmoid(torch.from_numpy(get_net_scores(network, val["X"]))).numpy()
p, r, t = precision_recall_curve(val["y"], val_probs)
f1 = 2*p*r/(p+r+1e-8) # Avoid division by zero
best_t = t[f1.argmax()]    
print(f"Best threshold on validation set: {best_t:.3f} (F1={f1.max():.3f})")

from plot import plot_roc_curve
plot_roc_curve(val["y"], val_probs, out_file_path="validation_roc_curve_val.png")

logits_test = get_net_scores(trained_network, test["X"])  
probs_test = torch.sigmoid(torch.from_numpy(logits_test)).numpy()

print(f"accuracy on test set: {accuracy_from_probs(probs_test, test['y'], thresh=best_t):.3f}")

keys_test = test["keys"]
# Print top 5 predictions
for i in np.argsort(-probs_test)[:5]:
    print(f"{keys_test[i]}: {probs_test[i]:.3f}  label={test['y'][i]}")

# Save the trained network scores
all_logits = get_net_scores(trained_network, np.concatenate([train["X"], val["X"], test["X"]]))
network_scores = {}
for k, v in zip(np.concatenate([train["keys"], val["keys"], test["keys"]]), all_logits):
    network_scores[k] = v

import pickle
with open("data/network_all_scores.pkl", "wb") as f:
    pickle.dump(network_scores, f)

test_logits = get_net_scores(trained_network, np.concatenate([val["X"], test["X"]]))
test_scores = {}
for k, v in zip(np.concatenate([val["keys"], test["keys"]]), test_logits):
    test_scores[k] = v

with open("data/network_test_scores.pkl", "wb") as f:
    pickle.dump(test_scores, f)