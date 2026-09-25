from fairfl.core.registry import register_strategy
from fairfl.strategies.base import Strategy


@register_strategy("fedavg")
class FedAvg(Strategy):
    """McMahan et al., AISTATS 2017: sample-size weighted mean."""
