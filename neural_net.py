import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt



class SimpleDenseNet(nn.Module):
    """
    Very small fully-connected classifier for sequence ESM embeddings.

    Architecture
    ------------
    [emb_dim] -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> logit
    """
    def __init__(self, esm_emb_dim: int = 1280, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(esm_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, emb_dim] containing float32 embeddings.

        Returns
        -------
        torch.Tensor
            Shape [batch] – raw logits (unnormalised scores).
        """
        return self.net(x).squeeze(1)


def prepare_loader(x, y, batch_size=64):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def prepare_train_loader(x, y, n_neg, n_pos, batch_size=64):
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
    ds = TensorDataset(x, y)

    # probability = 1 / class frequency
    class_w = torch.tensor([1/n_neg, 1/n_pos])
    sample_w = class_w[y.long()]
    from torch.utils.data import WeightedRandomSampler
    sampler   = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    loader = DataLoader(ds, batch_size=64, sampler=sampler)
    return loader


def train_net(model,
              loader,
              num_epochs: int = 10,
              lr: float = 1e-3,
              ):
    """
    Train the dense network on positive / negative ESM embeddings.
    Returns
    -------
    SimpleDenseNet – trained model.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_pos, n_neg = 219, 2664  
    pos_weight = torch.tensor([n_neg / n_pos])   # ≈ 12.17
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optim.zero_grad()
            logits = model(xb)
            loss = loss_function(logits, yb)
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)

        avg = running_loss / len(loader.dataset)
        losses.append(avg)
        print(f"Epoch {epoch + 1:02d}/{num_epochs} – loss: {avg:.4f}")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig("training_loss.png")
    return model


@torch.no_grad()
def get_net_scores(trained_net,
                   esm_seq_embeddings,
                   ):
    """
    Compute logits for a batch of embeddings using a trained SimpleDenseNet.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_net.eval().to(device)
    x = torch.as_tensor(np.asarray(esm_seq_embeddings),
                        dtype=torch.float32, device=device)
    logits = trained_net(x).cpu().numpy()
    return logits