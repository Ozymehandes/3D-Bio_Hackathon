import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1)  # no sigmoid -> we use BCEWithLogitsLoss
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


def prepare_loader(pos_emb, neg_emb, batch_size=64):
    """Create a PyTorch DataLoader from positive & negative numpy arrays."""
    x = np.vstack([pos_emb, neg_emb]).astype("float32")
    y = np.hstack([np.ones(len(pos_emb)), np.zeros(len(neg_emb))]).astype("float32")

    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


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
    loss_function = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
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
        print(f"Epoch {epoch + 1:02d}/{num_epochs} – loss: {avg:.4f}")

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