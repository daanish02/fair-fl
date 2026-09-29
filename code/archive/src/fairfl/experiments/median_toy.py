"""Chen et al. (NeurIPS 2020) Fig. 1: three nodes with f_i(x) = (x - a_i)^2 / 2, a = (1, 2, 10), step 0.001,
x_0 = 0.0005. The coordinate median of the local gradients (medianSGD) and signSGD with majority vote both drive the
*median* gradient to 0 and stall near x = a_2 = 2, while the true (mean) gradient stays at about 7/3. Adding symmetric
noise to each local gradient before the median (Algorithm 4) moves the expected median towards the mean."""

from __future__ import annotations

import json

import numpy as np

A = np.array([1.0, 2.0, 10.0])


def run(method: str, b: float = 0.0, steps: int = 10_000, lr: float = 1e-3, x0: float = 5e-4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = x0
    for _ in range(steps):
        g = x - A + (b * rng.standard_normal(3) if b > 0 else 0.0)
        if method == "median":
            x -= lr * np.median(g)
        elif method == "signsgd":
            x -= lr * np.sign(np.sign(g).sum())
        elif method == "mean":
            x -= lr * g.mean()
        else:
            raise ValueError(method)
    return {"method": method, "b": b, "x": float(x), "mean_grad": float(np.mean(x - A)),
            "median_grad": float(np.median(x - A))}


def main() -> None:
    rows = [run("mean"), run("median"), run("signsgd")]
    rows += [run("median", b, steps=100_000) for b in (1.0, 5.0, 10.0, 20.0)]
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
