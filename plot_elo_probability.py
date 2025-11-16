import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from methods.utils import expected_score

def update_elo(rating_a, rating_b, score_a, K=40):
    exp_a = expected_score(rating_a, rating_b)
    return rating_a + K * (score_a - exp_a)

# --- Setup ---
K_values = [10, 20, 30, 40]
K_colors = {10: 'blue', 20: 'green', 30: 'orange', 40: 'red'}
diffs = np.arange(0, 751, 10)   # rating differences (a - b)
rating_b = 1000 - diffs          # fixed baseline
rating_a = 1000 + diffs          # vary player A

# --- Compute curves ---
probs = [expected_score(ra, rb) for ra, rb in zip(rating_a, rating_b)]
probs = np.array(probs)
# Elo changes for win/loss at different K
elo_changes = {}
for K in K_values:
    changes_win = [update_elo(ra, rb, 1, K) - ra for ra, rb in zip(rating_a, rating_b)]
    changes_loss = [abs(update_elo(ra, rb, 0, K) - ra) for ra, rb in zip(rating_a, rating_b)]
    changes_draw = [abs(update_elo(ra, rb, 0.5, K) - ra) for ra, rb in zip(rating_a, rating_b)]
    elo_changes[K] = (changes_win, changes_loss, changes_draw)

# --- Plot ---
fig, ax1 = plt.subplots(figsize=(12,5))

# Left axis: probability of winning
ax1.plot(2*diffs, probs, 'k-', marker='s', markevery=3, label="$P(A\succ B)$")
ax1.plot(2*diffs, 1-probs, 'k--', marker='s', markevery=3, label="$P(A\prec B)$")
ax1.set_xlabel("Rating Difference (Item A - Item B)", fontsize=16)
ax1.set_ylabel("Probability", color="black", fontsize=16)
ax1.tick_params(axis='y', labelcolor="black", labelsize=16)
ax1.tick_params(axis='x', labelcolor="black", labelsize=16)

# Add minor ticks (5 subdivisions per major)
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax1.grid(True, which="major", linestyle="solid", alpha=0.7)
ax1.grid(True, which="minor", linestyle="--", alpha=0.5)

# Right axis: Elo change
ax2 = ax1.twinx()
for K, (wins, losses, draws) in elo_changes.items():
    if K == 40:
        ax2.plot(2*diffs, wins, label=f"K={K} expected outcome", marker='o', markevery=5, color=K_colors[K])
        ax2.plot(2*diffs, losses, label=f"K={K} unexpected outcome", linestyle=":", marker='o', markevery=5, color=K_colors[K])
    else:
        ax2.plot(2*diffs, wins, label=f"K={K}", marker='o', markevery=5, color=K_colors[K])
        ax2.plot(2*diffs, losses, label=f"K={K}", linestyle=":", marker='o', markevery=5, color=K_colors[K])
ax2.set_ylabel("Elo Change", color="blue", fontsize=16)
ax2.tick_params(axis='y', labelcolor="blue", labelsize=14)
# Add minor ticks for right axis too
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
# Formatting
ax1.set_xlim(0, 1500)
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 50)
ax1.legend(loc="upper left", fontsize=16, ncol=1, bbox_to_anchor=(-0.05, 1.35))
ax2.legend(loc="upper center", fontsize=16, ncol=4, bbox_to_anchor=(0.6, 1.35))
plt.tight_layout()
plt.savefig("figures/elo_probability_vs_rating_difference.pdf", dpi=600)
plt.close()