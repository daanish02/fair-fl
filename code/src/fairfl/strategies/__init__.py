"""Importing this package registers every strategy with registry.py's registry."""

import fairfl.clients  # noqa: F401  (sets Strategy.client_cls default, before any strategy imports a client)

from fairfl.strategies import fairrfl, fairweight, fcfl, fedavg, fedcda, fedfdp, fedmut, logofair, median, signsgd  # noqa: F401,E402
