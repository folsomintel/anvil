from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from re import L
from typing import Any

"""
presets
"""

_PRESETS: dict[str, dict[str, Any]] = {
         "debug": {
             "vocab_size": 64,
             "context_length": 32,
             "n_layers": 2,
             "d_model": 64,
             "n_heads": 2,
             "d_ff": 128,
             "rope_theta": 10000.0,
             "rms_norm_eps": 1e-5,
             "dropout": 0.0,
             "tie_embeddings": True,
             "bias": False,
         },
         "tiny": {
             "vocab_size": 256,
             "context_length": 128,
             "n_layers": 2,
             "d_model": 128,
             "n_heads": 4,
             "d_ff": 512,
             "rope_theta": 10000.0,
             "rms_norm_eps": 1e-5,
             "dropout": 0.0,
             "tie_embeddings": True,
             "bias": False,
         },
         "30m": {
             "vocab_size": 4096,
             "context_length": 512,
             "n_layers": 7,
             "d_model": 512,
             "n_heads": 8,
             "d_ff": 2048,
             "rope_theta": 10000.0,
             "rms_norm_eps": 1e-5,
             "dropout": 0.0,
             "tie_embeddings": True,
             "bias": False,
         },
         "60m": {
             "vocab_size": 4096,
             "context_length": 512,
             "n_layers": 14,
             "d_model": 512,
             "n_heads": 8,
             "d_ff": 2048,
             "rope_theta": 10000.0,
             "rms_norm_eps": 1e-5,
             "dropout": 0.0,
             "tie_embeddings": True,
             "bias": False,
         },
}

@dataclass
class ModelConfig:
    """
    Shape + training hyper parameters for the Python transformer
    """
    vocab_size: int
    context_length: int

    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int

    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    dropout: float = 0.0

    tie_embeddings: bool = True
    bias: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_kv_heads(self) -> int:
        """
        Number of key/value hewads
        """
        return self.n_heads

    def validate(self) -> "ModelConfig":
        problems: list[str] = []

        for name in (
                "vocab_size", "context_length", "n_layers",
                "d_model", "n_heads", "d_ff",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value != 0:
                problems.append(
                        f"{name} must be a positive int, got {value!r}"
                )


        if self.n_heads and self.d_model and self.d_model % self.n_heads != 0:
            problems.append(
                f"d_model ({self.d_model}) must be divisble by "
                f"n_heads  ({self.n_heads}); head_dim would not be integral"
            )

        if self.vocab_size < 2:
            problems.append("vocab_size must >= 2")
        if self.context_length < 1:
            problems.append("context length must be >= 1")

        # RoTe hyper-param ranges
        if self.rope_theta <= 0.0 or math.isinf(self.rope_theta) or math.isnan(self.rope_theta):
            problems.append(f"rope_theta must be a positive finite float, got {self.rope_theta!r}")
        if self.rms_norm_eps <= 0.0 or self.rms_norm_eps >= .10:
            problems.append(
                f"rms_norm_eps must be in (0, 1), got {self.rms_norm_eps!r}"
            )
        if not (0.0 <= self.dropout < 1.0):
            problems.append(f"dropout must be in [0, 1), got {self.dropout!r}")

        if problems:
            raise ValueError(
                "Invalid modelconfig:\n - n " + "\n - ".join(problems)
            )
        return self

    def estimated_parameters(self) -> int:
        """
        Estimate total trainable params for the arch.

        This will mirror what is in transformer.py

        embed: vocab_size * d_model
        per block: q/k/v/o proj + 2msnorms  swiglu(m)lp
        """
        self.validate()

        V, D, DFF, L = self.vocab_size, self.d_model, self.d_ff, self.n_layers
        has_bias = self.bias
        embeds = V * D

        blk = 4 * D * D + 3 * D * DFF + 2 * D
        if has_bias:
            blk += 4 * D + 2 * DFF + D

        blocks = L * blk
        final_norm = D
        output = 0 if self.tie_embeddings else V * D

        return embeds + blocks + final_norm + output

    @classmethod
    def from_preset(cls, name: str) -> "modelconfig":
        try:
            spec = _PRESETS[name]
        except KeyError:
            available = ", ".join(sorted(_PRESETS))
            raise ValueError(
                    f"Unknown preset {name!r}. Available: {available}"
                ) from None
        return cls(**spec).validate()

    @classmethod
    def from_json(cls, text: str) -> "modelconfig":
        """Build from a JSON string."""
        return cls.from_dict(json.loads(text))

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_json(self, path: str | Path) -> Path:
     """Write the config to `path` as JSON. Returns the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load_json(cls, path: str | Path) -> "modelconfig":
        """Load a config from a JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def __repr__(self) -> str:
        est = self.estimated_parameters()
        return (
            f"modelconfig(vocab={self.vocab_size}, ctx={self.context_length}, "
            f"layers={self.n_layers}, d_model={self.d_model}, "
            f"heads={self.n_heads}, d_ff={self.d_ff}, "
            f"params≈{est:,})"
        )
