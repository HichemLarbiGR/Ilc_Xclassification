"""
Multiple Instance Learning Models
=================================

This module implements Multiple Instance Learning (MIL) architectures
for necrosis score prediction from slide embeddings.

Models
------
SimpleMIL
    MLP encoder with mean pooling and no attention mechanism.
MeanMaxPoolMIL
    MLP encoder with configurable mean or max pooling.
AttentionMIL
    Gated attention MIL architecture inspired by Ilse et al. (2018).
TransMIL
    Transformer-based MIL architecture with a learnable CLS token.

Utilities
---------
build_model
    Instantiate a MIL model from a model identifier.
EarlyStopping
    Monitor validation loss and track the early stopping condition.

Notes
-----
Input bags may contain different numbers of patch embeddings.
Each model processes bags independently before generating
slide-level predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMIL(nn.Module):
    """
    Implement a basic MIL model without an attention mechanism.

    Patch embeddings are normalized and projected into a hidden
    representation. Mean pooling aggregates patch representations
    before the classification layer.

    Parameters
    ----------
    in_dim : int, default=1536
        Input patch embedding dimension.
    hidden_dim : int, default=128
        Hidden representation dimension.
    num_classes : int, default=2
        Number of output classes.
    dropout : float, default=0.1
        Dropout probability used in the encoder.
    """

    def __init__(
        self,
        in_dim=1536,
        hidden_dim=128,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(in_dim)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, bags):
        """
        Compute slide-level predictions for a batch of bags.

        Parameters
        ----------
        bags : list[torch.Tensor]
            List of tensors with shape (n_patches, in_dim).

        Returns
        -------
        tuple
            Logits of shape (B, num_classes), None, and a list
            of None values for compatibility with the existing pipeline.
        """
        bag_vecs = []

        for emb in bags:
            emb_norm = self.norm(emb)
            h = self.encoder(emb_norm)
            z = h.mean(dim=0)
            bag_vecs.append(z)

        z = torch.stack(bag_vecs)
        logits = self.classifier(z)

        return logits, None


class MeanMaxPoolMIL(nn.Module):
    """
    Implement a baseline MIL model using mean or max pooling.

    Patch embeddings are projected through an MLP and aggregated
    into a slide-level representation before classification.

    Parameters
    ----------
    in_dim : int, default=1536
        Input patch embedding dimension.
    hidden_dim : int, default=256
        Hidden representation dimension.
    num_classes : int, default=2
        Number of output classes.
    pool : str, default="mean"
        Aggregation strategy: "mean" or "max".
    """

    def __init__(
        self,
        in_dim=1536,
        hidden_dim=256,
        num_classes=2,
        pool="mean",
    ):
        super().__init__()

        assert pool in ("mean", "max"), (
            "pool must be either 'mean' or 'max'."
        )
        self.pool = pool

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, bags):
        """
        Compute slide-level predictions using the selected pooling method.

        Parameters
        ----------
        bags : list[torch.Tensor]
            List of tensors with shape (n_patches, in_dim).

        Returns
        -------
        tuple
            Logits of shape (B, num_classes) and None.
        """
        bag_vecs = []

        for emb in bags:
            h = self.encoder(emb)

            if self.pool == "mean":
                z = h.mean(dim=0)
            else:
                z = h.max(dim=0).values

            bag_vecs.append(z)

        z = torch.stack(bag_vecs)
        logits = self.classifier(z)

        return logits, None


class AttentionMIL(nn.Module):
    """
    Implement a gated attention-based MIL model.

    Patch embeddings are normalized and projected into a hidden space.
    Gated attention scores weight the patch representations to form
    a slide-level representation for classification.

    Parameters
    ----------
    in_dim : int, default=1536
        Input patch embedding dimension.
    hidden_dim : int, default=64
        Hidden representation dimension.
    attn_dim : int, default=32
        Attention projection dimension.
    num_classes : int, default=2
        Number of output classes.
    dropout : float, default=0.3
        Dropout probability used in the encoder.
    """

    def __init__(
        self,
        in_dim=1536,
        hidden_dim=64,
        attn_dim=32,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(in_dim)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

        self.head = nn.Linear(hidden_dim, num_classes)

    def attention(self, h):
        """
        Compute normalized gated attention weights.

        Parameters
        ----------
        h : torch.Tensor
            Encoded patch representations of shape (n_patches, hidden_dim).

        Returns
        -------
        torch.Tensor
            Attention weights of shape (n_patches,).
        """
        a = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))
        a = self.attn_w(a)

        return F.softmax(a, dim=0).squeeze(-1)

    def forward(self, bags):
        """
        Compute slide-level predictions and patch attention weights.

        Parameters
        ----------
        bags : list[torch.Tensor]
            List of tensors with shape (n_patches, in_dim).

        Returns
        -------
        tuple
            Logits of shape (B, num_classes) and a list of detached
            attention-weight tensors, one per bag.
        """
        bag_vecs = []
        attn_list = []

        for emb in bags:
            emb_norm = self.norm(emb)
            h = self.encoder(emb_norm)

            a = self.attention(h)
            z = (a.unsqueeze(-1) * h).sum(dim=0)

            bag_vecs.append(z)
            attn_list.append(a.detach())

        z = torch.stack(bag_vecs)
        logits = self.head(z)

        return logits, attn_list


class TransMIL(nn.Module):
    """
    Implement a Transformer-based MIL model with a learnable CLS token.

    Patch embeddings are projected into the Transformer hidden space.
    A CLS token is prepended to each bag, and the Transformer encoder
    produces a slide-level representation for classification.

    Parameters
    ----------
    in_dim : int, default=512
        Input patch embedding dimension.
    d_model : int, default=256
        Transformer hidden dimension, divisible by n_heads.
    n_heads : int, default=4
        Number of attention heads.
    n_layers : int, default=2
        Number of Transformer encoder layers.
    ffn_dim : int, default=512
        Feed-forward network dimension.
    num_classes : int, default=4
        Number of output classes.
    dropout : float, default=0.1
        Dropout probability used in the projection and encoder.

    Notes
    -----
    Self-attention has quadratic memory complexity with respect to
    the number of tokens. Large bags may therefore require patch sampling.
    """

    def __init__(
        self,
        in_dim=512,
        d_model=256,
        n_heads=4,
        n_layers=2,
        ffn_dim=512,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, bags):
        """
        Compute slide-level predictions using the CLS representation.

        Bags are processed independently, so no padding is required.

        Parameters
        ----------
        bags : list[torch.Tensor]
            List of tensors with shape (n_patches, in_dim).

        Returns
        -------
        tuple
            Logits of shape (B, num_classes) and None.
        """
        cls_outs = []

        for emb in bags:
            x = self.proj(emb).unsqueeze(0)
            cls = self.cls_token.expand(1, 1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.transformer(x)

            cls_outs.append(x[0, 0])

        z = torch.stack(cls_outs)
        logits = self.classifier(z)

        return logits, None


def build_model(
    name: str,
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a MIL model from its name.

    Parameters
    ----------
    name : str
        Model identifier.
    num_classes : int, default=2
        Number of output classes.
    **kwargs
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    nn.Module
        Instantiated MIL model.

    Raises
    ------
    ValueError
        If the model name is not recognized.

    Examples
    --------
    >>> model = build_model("attention", num_classes=4)
    >>> model = build_model("mean", num_classes=4, pool="max")
    >>> model = build_model("transformer", num_classes=4, n_layers=3)
    """
    name = name.lower()

    if name in ("mean", "meanpool", "max"):
        pool = "max" if name == "max" else kwargs.pop("pool", "mean")

        return MeanMaxPoolMIL(
            num_classes=num_classes,
            pool=pool,
            **kwargs,
        )

    if name in ("attention", "attn", "abmil"):
        return AttentionMIL(num_classes=num_classes, **kwargs)

    if name in ("transformer", "transmil", "trans"):
        return TransMIL(num_classes=num_classes, **kwargs)

    raise ValueError(
        f"Unknown model name: {name!r}. "
        "Choose from 'mean', 'attention', 'transformer'."
    )


class EarlyStopping:
    """
    Track validation loss to determine when training should stop.

    Parameters
    ----------
    patience : int, default=20
        Number of consecutive non-improving epochs allowed.
    min_delta : float, default=0.001
        Minimum required decrease in validation loss to qualify
        as an improvement.

    Attributes
    ----------
    counter : int
        Number of consecutive non-improving epochs.
    best_loss : float or None
        Best validation loss observed so far.
    early_stop : bool
        Whether the stopping criterion has been reached.
    """

    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Update the early stopping state using the current validation loss.

        Parameters
        ----------
        val_loss : float
            Validation loss for the current epoch.
        """
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0
