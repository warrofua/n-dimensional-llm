"""Mutual information proxy based on InfoNCE with lightweight heads."""
from __future__ import annotations

from typing import Any, Tuple

try:  # pragma: no cover - torch is optional at import time
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]
    F = Any  # type: ignore[assignment]


class MIProxy(nn.Module):
    """InfoNCE-style proxy estimating a lower bound on mutual information.

    The architecture mirrors the lightweight twin-projector design outlined in
    the whitepaper, consisting of two shallow MLPs (:math:`f` and :math:`h`) and
    a temperature-scaled cosine similarity classifier.  The module accepts either
    token-level tensors of shape ``(batch, tokens, dim)`` or pre-pooled
    representations of shape ``(batch, dim)`` and supports both CPU and CUDA
    tensors transparently.
    """

    def __init__(self, d_model: int, d_proj: int = 256, temperature: float = 0.07) -> None:
        if torch is None:  # pragma: no cover - defensive guard when torch missing
            raise RuntimeError("torch is required to instantiate MIProxy")
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_proj <= 0:
            raise ValueError("d_proj must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.f = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, d_proj),
        )
        self.h = nn.Sequential(
            nn.Linear(d_model, d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, d_proj),
        )
        self.tau = float(temperature)

    def forward(self, z: Tensor, y_repr: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D401 - standard module contract
        """Compute the InfoNCE lower bound and classifier logits.

        Parameters
        ----------
        z:
            Either a tensor of shape ``(batch, tokens, dim)`` containing the
            selected token embeddings per sample, or a pre-pooled tensor of
            shape ``(batch, dim)``.  If token-level inputs are supplied they are
            mean-pooled internally, closely matching the reference design in the
            whitepaper.
        y_repr:
            Tensor of shape ``(batch, dim)`` representing the target/label
            embeddings associated with each sample.

        Returns
        -------
        Tuple[Tensor, Tensor]
            ``(mi_lower_bound, logits)`` where ``mi_lower_bound`` is the
            InfoNCE lower bound (negative cross-entropy) and ``logits`` are the
            temperature-scaled similarity scores.
        """

        if torch is None:  # pragma: no cover - defensive when torch absent
            raise RuntimeError("torch is required to evaluate MIProxy")
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z)
        if not isinstance(y_repr, torch.Tensor):
            y_repr = torch.as_tensor(y_repr)

        if z.dim() == 2:
            pooled = z
        elif z.dim() == 3:
            pooled = z.mean(dim=1)
        else:
            raise ValueError("z must have shape (batch, dim) or (batch, tokens, dim)")

        if pooled.size(0) != y_repr.size(0):
            raise ValueError("batch dimension mismatch between z and y_repr")

        param = next(self.parameters(), None)
        if param is None:  # pragma: no cover - module always has parameters
            device = pooled.device
            dtype = pooled.dtype
        else:
            device = param.device
            dtype = param.dtype

        pooled = pooled.to(device=device, dtype=dtype, copy=False)
        targets = y_repr.to(device=device, dtype=dtype, copy=False)

        pooled = F.normalize(self.f(pooled), dim=-1)
        target = F.normalize(self.h(targets), dim=-1)
        logits = pooled @ target.transpose(0, 1)
        logits = logits / self.tau
        labels = torch.arange(pooled.size(0), device=device)
        loss = F.cross_entropy(logits, labels)
        return -loss, logits


__all__ = ["MIProxy"]
