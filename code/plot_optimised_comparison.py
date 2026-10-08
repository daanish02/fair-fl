# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "seaborn",
#     "pandas",
# ]
# ///
"""Compare centralised vs 10-client federated (optimised run) with Seaborn plots."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent / "results" / "optimised"
OUT_DIR = Path(__file__).parent / "results" / "figures" / "optimised"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", palette="deep")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

CENTRALISED = json.load(open(RESULTS_DIR / "centralised_seed0.json"))
FEDERATED = json.load(open(RESULTS_DIR / "federated_c10_a0.5_seed0.json"))

COLORS = {"Centralised": "#4C72B0", "Federated (10 clients)": "#DD8452"}


def history_df(d, label):
    df = pd.DataFrame(d["history"])
    df["setting"] = label
    return df


hist = pd.concat(
    [history_df(CENTRALISED, "Centralised"), history_df(FEDERATED, "Federated (10 clients)")],
    ignore_index=True,
)

# 1. Test accuracy vs round
fig, ax = plt.subplots(figsize=(9, 6))
sns.lineplot(data=hist, x="round", y="test_acc", hue="setting", marker="o", linewidth=2.5, ax=ax, palette=COLORS, errorbar=None)
ax.set_title("Test Accuracy over Training Rounds", fontweight="bold")
ax.set_xlabel("Round")
ax.set_ylabel("Test Accuracy")
ax.legend(title=None)
sns.despine()
fig.savefig(OUT_DIR / "accuracy_over_rounds.png")
plt.close(fig)

# 2. Train loss vs round
fig, ax = plt.subplots(figsize=(9, 6))
sns.lineplot(data=hist, x="round", y="train_loss", hue="setting", marker="o", linewidth=2.5, ax=ax, palette=COLORS, errorbar=None)
ax.set_title("Training Loss over Rounds", fontweight="bold")
ax.set_xlabel("Round")
ax.set_ylabel("Train Loss")
ax.legend(title=None)
sns.despine()
fig.savefig(OUT_DIR / "loss_over_rounds.png")
plt.close(fig)

# 3. Elapsed time per round
fig, ax = plt.subplots(figsize=(9, 6))
sns.lineplot(data=hist, x="round", y="elapsed_s", hue="setting", marker="o", linewidth=2.5, ax=ax, palette=COLORS, errorbar=None)
ax.set_title("Wall-clock Time per Round", fontweight="bold")
ax.set_xlabel("Round")
ax.set_ylabel("Elapsed Time (s)")
ax.legend(title=None)
sns.despine()
fig.savefig(OUT_DIR / "time_per_round.png")
plt.close(fig)

# 4. Per-client accuracy (fairness) - federated clients vs centralised single value
client_rows = []
for cid, acc in FEDERATED["client_acc"].items():
    client_rows.append({"client": f"C{cid}", "accuracy": acc, "setting": "Federated (10 clients)"})
for cid, acc in CENTRALISED["client_acc"].items():
    client_rows.append({"client": "Global", "accuracy": acc, "setting": "Centralised"})
client_df = pd.DataFrame(client_rows)

fig, ax = plt.subplots(figsize=(10, 6))
fed_df = client_df[client_df["setting"] == "Federated (10 clients)"].sort_values("client")
sns.barplot(data=fed_df, x="client", y="accuracy", color=COLORS["Federated (10 clients)"], ax=ax, label="Federated clients")
cent_acc = CENTRALISED["client_acc"]["0"]
ax.axhline(cent_acc, color=COLORS["Centralised"], linestyle="--", linewidth=2.5, label=f"Centralised ({cent_acc:.3f})")
mean_fed = fed_df["accuracy"].mean()
ax.axhline(mean_fed, color="#55A868", linestyle=":", linewidth=2.5, label=f"Federated mean ({mean_fed:.3f})")
ax.set_title("Per-Client Final Accuracy: Fairness Comparison", fontweight="bold")
ax.set_xlabel("Client")
ax.set_ylabel("Final Test Accuracy")
ax.legend()
sns.despine()
fig.savefig(OUT_DIR / "per_client_accuracy_fairness.png")
plt.close(fig)

# 5. Accuracy distribution spread (box + strip) for federated clients only
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=fed_df, y="accuracy", color=COLORS["Federated (10 clients)"], width=0.35, ax=ax)
sns.stripplot(data=fed_df, y="accuracy", color="black", size=8, jitter=0.08, ax=ax)
ax.axhline(cent_acc, color=COLORS["Centralised"], linestyle="--", linewidth=2.5, label=f"Centralised ({cent_acc:.3f})")
ax.set_title("Federated Client Accuracy Spread vs Centralised", fontweight="bold")
ax.set_ylabel("Final Test Accuracy")
ax.legend()
sns.despine()
fig.savefig(OUT_DIR / "accuracy_spread_fairness.png")
plt.close(fig)

print("Saved figures to", OUT_DIR)
