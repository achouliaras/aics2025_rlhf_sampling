import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from methods.utils import expected_score

# --- Tie model ---
def tie_probability(skill_diff, max_tie_prob=0.2, tau=200):
    """Tie probability decays exponentially with skill difference."""
    return max_tie_prob * np.exp(-skill_diff / tau)

# --- Setup ---
diffs = np.arange(0, 1501, 10)   # skill differences
rating_a = 1000 + diffs
rating_b = 1000 - diffs

# Probabilities from Elo
p_win_base = np.array([expected_score(ra, rb) for ra, rb in zip(rating_a, rating_b)])
p_tie = tie_probability(diffs, max_tie_prob=0.33, tau=200)

# Rescale win/loss so win + loss = 1 - tie
p_win = (1 - p_tie) * p_win_base
p_loss = (1 - p_tie) * (1 - p_win_base)

# --- Plot as stacked bands ---
fig, ax = plt.subplots(figsize=(6, 3))

ax.stackplot(
    diffs*2, 
    p_loss, p_tie, p_win, 
    labels=["Item B preference", "Equal preference", "Item A preference"],
    colors=["#f1948a", "#85c1e9", "#82e0aa"],
    alpha=0.8
)

ax.set_xlabel("Value Difference (Item A - Item B)", fontsize=16)
ax.set_ylabel("Probability", fontsize=16)
# ax.set_title("Outcome Probabilities vs Value Difference", fontsize=18)

ax.set_xlim(0, 1500)
ax.set_ylim(0, 1)
ax.tick_params(axis='y', labelcolor="black", labelsize=14)
ax.tick_params(axis='x', labelcolor="black", labelsize=14)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax.grid(True, which="major", linestyle="solid", alpha=0.7)
ax.grid(True, which="minor", linestyle="--", alpha=0.5)

ax.legend(loc="upper right", fontsize=16)
plt.tight_layout()
plt.savefig("figures/elo_win_tie_loss_bands.pdf", dpi=600)
plt.close()
