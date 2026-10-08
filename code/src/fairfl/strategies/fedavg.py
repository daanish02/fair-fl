from __future__ import annotations

from fairfl.registry import register_strategy
from fairfl.strategies.base import Strategy


@register_strategy("fedavg")
class FedAvg(Strategy):
    """Sample-size-weighted average of client updates. Every hook: inherited default."""
