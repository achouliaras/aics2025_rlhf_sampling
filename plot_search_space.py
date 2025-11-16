import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes
from methods.swiss_elo import simulate_swiss_tournament_elo


def run_swiss_eval(N, initial_rating, num_seeds=100, max_rounds=30):
    results = []
    games_counts = []

    for seed in range(num_seeds):
        random.seed(seed)
        np.random.seed(seed)

        ratings0 = {f"Player_{i+1}": initial_rating for i in range(N)}
        true_skills = {f"Player_{i+1}": random.gauss(initial_rating, 200) for i in range(N)}

        # run swiss
        final_ratings, games = simulate_swiss_tournament_elo(
            N, true_skills, ratings0.copy(), k=1, max_rounds=max_rounds, debug=False
        )

        df = pd.DataFrame({
            "Player": list(true_skills.keys()),
            "TrueSkill": list(true_skills.values()),
            "Swiss Tournament": list(final_ratings.values())
        })

        corr = df["TrueSkill"].corr(df["Swiss Tournament"])
        results.append(corr)
        games_counts.append(games)

    mean_corr = np.mean(results)
    std_corr = np.std(results, ddof=1)
    ci_halfwidth = 1.96 * std_corr / np.sqrt(num_seeds)  # 95% CI

    mean_games = np.mean(games_counts)
    std_games = np.std(games_counts, ddof=1)

    return {
        "mean_corr": mean_corr,
        "lower_ci": mean_corr - ci_halfwidth,
        "upper_ci": mean_corr + ci_halfwidth,
        "mean_games": mean_games,
        "std_games": std_games
    }

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    # --- Parameters ---
    N = 100
    K = 40
    initial_rating = 1000
    num_seeds = 100
    min_rounds = 5
    max_rounds = 200
    
    # --- Run Swiss separately ---
    swiss_summary = run_swiss_eval(N, initial_rating, num_seeds=num_seeds, max_rounds=30)
    print("Swiss Tournament (30 rounds)")
    print(f"Correlation: {swiss_summary['mean_corr']:.3f} "
          f"[{swiss_summary['lower_ci']:.3f}, {swiss_summary['upper_ci']:.3f}]")
    print(f"Games: {swiss_summary['mean_games']:.1f} ± {swiss_summary['std_games']:.1f}")

    # load results from CSV

    if not os.path.exists("src/swiss_infogain_results_agg.csv"):
        raise FileNotFoundError("swiss_infogain_results_agg.csv not found. Please run the simulations first.")
    df_swiss = pd.read_csv("src/swiss_infogain_results_agg.csv")    
    if not os.path.exists("src/round_robin_results_agg.csv"):
        raise FileNotFoundError("round_robin_results_agg.csv not found. Please run the simulations first.")
    df_agg = pd.read_csv("src/round_robin_results_agg.csv")    
    if not os.path.exists("src/search_space_comparison.csv"):
        raise FileNotFoundError("search_space_comparison.csv not found. Please run the simulations first.")
    df = pd.read_csv("src/search_space_comparison.csv")
    methods = ["full_rng_elo","swiss_elo", "borda", "bt"]
    x = df["Rounds"] * N  # Convert rounds to games
    titles={
        "full_rng_elo": "Elo-RND",
        "swiss_elo": "Elo RND+Swiss",
        "borda": "Borda-RND",
        "bt": "Bradley-Terry",
        "Elo-RR": "Elo-Copeland",
        "Borda-RR": "Borda-Copeland"
    }
    # Main plot
    fig, ax = plt.subplots(figsize=(10,5))
    for m in methods:
        ax.plot(x, df[f"{m}_mean"], label=titles[m])
        ax.fill_between(x, df[f"{m}_lower_ci"], df[f"{m}_upper_ci"], alpha=0.2)
    ax.plot(df_swiss["AvgGames_mean"], df_swiss["SwissInfoGain_mean"], label="Swiss InfoGain", linestyle='-', color='cyan', marker='o', markersize=4)
    # ax.axhline(y=df_swiss["SwissInfoGain_mean"].iloc[-1], color="cyan", linestyle="-", linewidth=2, alpha=0.7)
    ax.fill_between(df_swiss["AvgGames_mean"], df_swiss["SwissInfoGain_lower_ci"], df_swiss["SwissInfoGain_upper_ci"], color='cyan', alpha=0.2)
    for m in ["Elo-RR", "Borda-RR"]:
        ax.plot(df_agg["Rounds"]*4950, df_agg[f"{m}_mean"], label=titles[m], linestyle='--', marker='o', markersize=4)
    ax.plot(
        swiss_summary["mean_games"]+50, swiss_summary["mean_corr"],
        'o', linestyle='-', color='black', label='Swiss Tournament'
    )
    ax.axhline(y=swiss_summary["mean_corr"], color="black", linestyle="-", alpha=0.5)
    ax.axhspan(swiss_summary["lower_ci"], swiss_summary["upper_ci"], color="black", alpha=0.2)
    ax.set_xlabel("Number of comparisons", fontsize=16)
    ax.set_ylabel("Correlation", fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlim(450, 20000)
    ax.set_ylim(0.7, 1)
    ax.legend(loc='lower right', ncols=2, fontsize=12)
    ax.grid(True)

    # --- Add zoomed inset ---
    axins = zoomed_inset_axes(ax,
                              zoom=1.5,
                              loc='center', # anchor corner of the bbox
                              bbox_to_anchor=(0.235, 0.325),
                              bbox_transform=ax.transAxes, # interpret bbox_to_anchor in ax coordinates
                              borderpad=0
                             )

    for m in methods:
        axins.plot(x, df[f"{m}_mean"])
        axins.fill_between(x, df[f"{m}_lower_ci"], df[f"{m}_upper_ci"], alpha=0.2)
    axins.plot(df_swiss["AvgGames_mean"], df_swiss["SwissInfoGain_mean"], label="Swiss InfoGain", linestyle='-', color='cyan', marker='o', markersize=4)
    axins.fill_between(df_swiss["AvgGames_mean"], df_swiss["SwissInfoGain_lower_ci"], df_swiss["SwissInfoGain_upper_ci"], color='cyan', alpha=0.2)
    axins.plot(
        swiss_summary["mean_games"]+50, swiss_summary["mean_corr"],
        'o', linestyle='-', color='black', label='Swiss Tournament'
    )
    axins.axhline(y=swiss_summary["mean_corr"], color="black", linestyle="-", alpha=0.5)
    axins.axhspan(swiss_summary["lower_ci"], swiss_summary["upper_ci"], color="black", alpha=0.2)
    # Specify the zoomed region
    x1, x2, y1, y2 = 495, 3000, 0.9, 1.00  # adjust these
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.tick_params(axis="both", which="both", labelsize=15)
    axins.grid(True)
    # Draw lines connecting inset to main plot
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

    axins2 = zoomed_inset_axes(ax,
                              zoom=1.5,
                              loc='center', # anchor corner of the bbox
                              bbox_to_anchor=(0.65, 0.6),
                              bbox_transform=ax.transAxes, # interpret bbox_to_anchor in ax coordinates
                              borderpad=0
                             )

    for m in methods:
        axins2.plot(x, df[f"{m}_mean"])
        axins2.fill_between(x, df[f"{m}_lower_ci"], df[f"{m}_upper_ci"], alpha=0.2)
    axins2.axhline(y=df_swiss["SwissInfoGain_mean"].iloc[-1], color="cyan", linestyle="-", linewidth=1, alpha=0.7)
    axins2.axhspan(df_swiss["SwissInfoGain_lower_ci"].iloc[-1], df_swiss["SwissInfoGain_upper_ci"].iloc[-1], color="cyan", alpha=0.2)
    for m in ["Elo-RR", "Borda-RR"]:
        axins2.plot(df_agg["Rounds"]*4950, df_agg[f"{m}_mean"], label=titles[m], linestyle='--', marker='o', markersize=4)
    axins.plot(
        swiss_summary["mean_games"]+50, swiss_summary["mean_corr"],
        'o', linestyle='-', color='black', label='Swiss Tournament'
    )
    axins2.axhline(y=swiss_summary["mean_corr"], color="black", linestyle="-", alpha=0.5)
    axins2.axhspan(swiss_summary["lower_ci"], swiss_summary["upper_ci"], color="black", alpha=0.2)
    # Specify the zoomed region
    x1, x2, y1, y2 = 4900, 10000, 0.93, 0.99  # adjust these
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.tick_params(axis="both", which="both", labelsize=15)
    axins2.grid(True)
    # Draw lines connecting inset to main plot
    mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.5")

    plt.tight_layout()
    plt.savefig("figures/search_space_comparison.pdf", dpi=600)
    
