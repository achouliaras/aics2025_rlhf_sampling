import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from methods.elo_random import simulate_elo_full
from methods.swiss_elo import simulate_rnd_swiss_elo, simulate_swiss_tournament_elo, simulate_rnd_swiss_elo_batch
from methods.borda_count import simulate_borda_count_rnd
from methods.bradley_terry_model import simulate_bt
from methods.elo_roundrobin import simulate_elo_roundrobin
from methods.borda_count_round_robin import simulate_borda_round_robin
from methods.quicksort_active import simulate_quicksort
from methods.swiss_infogain import simulate_swiss_infogain
import warnings

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    N = 100
    initial_rating = 1000
    
    # suppress warnings
    warnings.filterwarnings('ignore')
    #  seeds = range(1)

    # # Accumulators
    # accum = {
    #     "TrueSkill": np.zeros(N),
    #     "Elo-RND": np.zeros(N),
    #     "Elo RND+Swiss": np.zeros(N),
    #     "Swiss Tournament": np.zeros(N),
    #     "Borda-RND": np.zeros(N),
    #     "Bradley-Terry": np.zeros(N),
    #     "Elo-Copeland": np.zeros(N),
    #     "Borda-Copeland": np.zeros(N),
    # }
    # player_ids = [f"Player_{i+1}" for i in range(N)]
    # ratings = {pid: initial_rating for pid in player_ids}

    # for seed in seeds:
    #     rng = random.Random(seed)
    #     np.random.seed(seed)
    #     true_skills = {pid: rng.gauss(initial_rating, 200) for pid in player_ids}

    #     final_ratings_full = simulate_elo_full(N, true_skills, ratings, rounds=10, K=40, debug=False)
    #     final_ratings_min = simulate_rnd_swiss_elo(N, true_skills, ratings, rnd_rounds=5, rounds=5, K=40, debug=False)
    #     final_ratings_tournament = simulate_swiss_tournament_elo(N, true_skills, ratings, k=1, rounds=11, debug=False)
    #     final_ratings_borda = simulate_borda_count_rnd(N, true_skills, ratings, rounds=10, debug=False)
    #     final_ratings_bt = simulate_bt(N, true_skills, ratings, games_per_player=10, debug=False)
    #     final_ratings_rr = simulate_elo_roundrobin(N, true_skills, ratings, rounds=1, K=40, debug=False)
    #     final_ratings_borda_elo = simulate_borda_round_robin(N, true_skills, ratings, rounds=1, debug=False)

    #     for i, pid in enumerate(player_ids):
    #         accum["TrueSkill"][i] += true_skills[pid]
    #         accum["Elo-RND"][i] += final_ratings_full[pid]
    #         accum["Elo RND+Swiss"][i] += final_ratings_min[pid]
    #         accum["Swiss Tournament"][i] += final_ratings_tournament[pid]
    #         accum["Borda-RND"][i] += final_ratings_borda[pid]
    #         accum["Bradley-Terry"][i] += final_ratings_bt[pid]
    #         accum["Elo-Copeland"][i] += final_ratings_rr[pid]
    #         accum["Borda-Copeland"][i] += final_ratings_borda_elo[pid]

    # # Average results
    # averaged_df = pd.DataFrame({
    #     "Player": player_ids,
    #     "TrueSkill": accum["TrueSkill"] / len(seeds),
    #     "Elo-RND": accum["Elo-RND"] / len(seeds),
    #     "Elo RND+Swiss": accum["Elo RND+Swiss"] / len(seeds),
    #     "Swiss Tournament": accum["Swiss Tournament"] / len(seeds),
    #     "Borda-RND": accum["Borda-RND"] / len(seeds),
    #     "Bradley-Terry": accum["Bradley-Terry"] / len(seeds),
    #     "Elo-Copeland": accum["Elo-Copeland"] / len(seeds),
    #     "Borda-Copeland": accum["Borda-Copeland"] / len(seeds),
    # })

    # # averaged_df = averaged_df.sort_values(by="TrueSkill", ascending=False).reset_index(drop=True)
    # averaged_df.to_csv("final_avg_ratings.csv", index=False)

    # # ---- CORRELATIONS ----
    # corr_full = averaged_df["TrueSkill"].corr(averaged_df["Elo-RND"])
    # corr_min = averaged_df["TrueSkill"].corr(averaged_df["Elo RND+Swiss"])
    # corr_tournament = averaged_df["TrueSkill"].corr(averaged_df["Swiss Tournament"])
    # corr_bt = averaged_df["TrueSkill"].corr(averaged_df["Bradley-Terry"])
    # corr_borda = averaged_df["TrueSkill"].corr(averaged_df["Borda-RND"])
    # corr_rr = averaged_df["TrueSkill"].corr(averaged_df["Elo-Copeland"])
    # corr_borda_elo = averaged_df["TrueSkill"].corr(averaged_df["Borda-Copeland"])

    # print("Correlation between True Skill and Elo-RND:", corr_full)
    # print("Correlation between True Skill and Elo RND+Swiss:", corr_min)
    # print("Correlation of True Skill and Swiss Tournament:", corr_tournament)
    # print("Correlation of True Skill and Bradley-Terry model:", corr_bt)
    # print("Correlation between True Skill and #Wins (Borda):", corr_borda)
    # print("Correlation between True Skill and Elo-Copeland:", corr_rr)
    # print("Correlation between True Skill and #Wins (Borda-Copeland):", corr_borda_elo)

    # final_df = averaged_df
    
    ratings = {f"Player_{i+1}": initial_rating for i in range(N)}
    true_skills = {f"Player_{i+1}": random.gauss(initial_rating, 200) for i in range(N)}


    final_ratings_full = simulate_elo_full(N,true_skills, ratings, rounds=11, K=80, debug=False)
    final_ratings_rr = simulate_elo_roundrobin(N, true_skills, ratings, rounds=1, K=40, debug=True)
    final_ratings_tournament, total_games_swiss = simulate_swiss_tournament_elo(N, true_skills, ratings, k=1, max_rounds=30, debug=True)
    final_ratings_quick = simulate_quicksort(N, true_skills, ratings, rounds=1000, debug=True)
    final_ratings_min = simulate_rnd_swiss_elo(N, true_skills, ratings, rnd_rounds=5, rounds=6, K=40, debug=True)
    final_ratings_borda = simulate_borda_count_rnd(N, true_skills, ratings, rounds=11, debug=True)
    final_ratings_bt = simulate_bt(N, true_skills, ratings, games_per_player=11, debug=True)
    final_ratings_borda_elo = simulate_borda_round_robin(N, true_skills, ratings, rounds=1, debug=True)
    final_ratings_infogain, total_games_swissinfogain = simulate_swiss_infogain(N, true_skills, ratings, k=1, max_rounds=11, debug=True)

    # ---- SAVE RESULTS ----
    # Save final ratings to CSV
    final_df = pd.DataFrame({
        "Elo-RND": list(final_ratings_full.values()),
        "Elo-Copeland": list(final_ratings_rr.values()),
        "Swiss Tournament": list(final_ratings_tournament.values()),
        "Quicksort": list(final_ratings_quick.values()),
        "Player": list(true_skills.keys()),
        "TrueSkill": list(true_skills.values()),
        "Elo RND+Swiss": list(final_ratings_min.values()),
        "Borda-RND": list(final_ratings_borda.values()),
        "Bradley-Terry": list(final_ratings_bt.values()),
        "Borda-Copeland": list(final_ratings_borda_elo.values()),
        "Swiss Infogain": list(final_ratings_infogain.values()),
    })

    final_df = final_df.sort_values(by="TrueSkill", ascending=False).reset_index(drop=True)
    final_df.to_csv("final_ratings.csv", index=False)

    # ---- CORRELATIONS ----
    corr_full = final_df["TrueSkill"].corr(final_df["Elo-RND"])
    corr_rr = final_df["TrueSkill"].corr(final_df["Elo-Copeland"])
    corr_tournament = final_df["TrueSkill"].corr(final_df["Swiss Tournament"])
    corr_quick = final_df["TrueSkill"].corr(final_df["Quicksort"])
    corr_min = final_df["TrueSkill"].corr(final_df["Elo RND+Swiss"])
    corr_bt = final_df["TrueSkill"].corr(final_df["Bradley-Terry"])
    corr_borda = final_df["TrueSkill"].corr(final_df["Borda-RND"])
    corr_borda_elo = final_df["TrueSkill"].corr(final_df["Borda-Copeland"])
    corr_infogain = final_df["TrueSkill"].corr(final_df["Swiss Infogain"])

    # print("Correlation between True Skill and Elo-RND:", corr_full)
    # print("Correlation between True Skill and Elo-Copeland:", corr_rr)
    # print("Correlation of True Skill and Swiss Tournament:", corr_tournament)
    # print("Correlation between True Skill and Elo RND+Swiss:", corr_min)
    # print("Correlation of True Skill and Bradley-Terry model:", corr_bt)
    # print("Correlation between True Skill and #Wins (Borda):", corr_borda)
    # print("Correlation between True Skill and #Wins (Borda-Copeland):", corr_borda_elo)
    # print("Correlation between True Skill and Swiss Infogain:", corr_infogain)

    # Grid of scatter plots
    # ("Quicksort", "blue", corr_quick),
    methods = [
        ("Bradley-Terry", "red", corr_bt),
        ("Borda-RND", "green", corr_borda),
        ("Borda-Copeland", "brown", corr_borda_elo),
        ("Elo-RND", "blue", corr_full),
        ("Elo RND+Swiss", "orange", corr_min),
        ("Elo-Copeland", "purple", corr_rr),
        ("Swiss Tournament", "black", corr_tournament),
        ("Swiss Infogain", "cyan", corr_infogain),
    ]
    
    # Add combined as last "method"
    methods_with_combined = methods + [("All Methods Combined", None, 1.0)]

    n_methods = len(methods_with_combined)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols  # ceil division

    # --- Grid of subplots ---
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (ax, (method, color, corr)) in enumerate(zip(axes, methods_with_combined)):
        row, col = divmod(idx, ncols)
        if method == "All Methods Combined":
            # plot all methods together
            for m, c, _ in methods:
                ax.scatter(final_df["TrueSkill"], final_df[m], color=c, alpha=0.5, label=m)
            ax.plot([400, 1600], [400, 1600], 'k--', label="y=x")
            # ax.legend(fontsize="small")
        else:
            # individual method
            ax.scatter(final_df["TrueSkill"], final_df[method], color=color, alpha=0.7, label=method)
            ax.plot([400, 1600], [400, 1600], 'k--')
            ax.legend(fontsize=16)

        # Hide labels depending on position
        if col > 0:  # hide y labels for columns 2+
            ax.set_ylabel("")
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Elo", fontsize=16)

        if row < nrows - 1:  # hide x labels for all but bottom row
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("True Skill", fontsize=16)

        if method == "Aggregate View":
            ax.set_title(method, fontsize=16)
        else:
            ax.set_title(method+f" (Corr: {corr:.3f})", fontsize=16)
        ax.grid(True)
        
        ax.tick_params(axis='x', labelcolor="black", labelsize=13)
        ax.tick_params(axis='y', labelcolor="black", labelsize=13)
        
        if method == "All Methods Combined":
            ax.set_title(method, fontsize=16)
        else:
            ax.set_title(method+f" (Corr: {corr:.3f})", fontsize=16)
        ax.grid(True)

    # Remove any unused subplot slots
    for ax in axes[n_methods:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig("figures/elo_vs_trueskill_grid.png")
    plt.close()

    # Distribution of final ratings
    plt.figure(figsize=(10,6))
    plt.hist(list(true_skills.values()), bins=20, alpha=0.3, label="True skills")
    # plt.hist(list(final_ratings_full.values()), bins=20, alpha=0.3, label="ELO RND")
    # plt.hist(list(final_ratings_rr.values()), bins=20, alpha=0.3, label="ELO RR")
    # plt.hist(list(final_ratings_tournament.values()), bins=20, alpha=0.3, label="Swiss Tournament")
    # plt.hist(list(final_ratings_quick.values()), bins=20, alpha=0.3, label="Quicksort")
    plt.hist(list(final_ratings_min.values()), bins=20, alpha=0.3, label="ELO RND+Swiss")
    plt.hist(list(final_ratings_bt.values()), bins=20, alpha=0.3, label="Bradley-Terry")
    plt.hist(list(final_ratings_borda.values()), bins=20, alpha=0.3, label="Borda-RND")
    plt.hist(list(final_ratings_borda_elo.values()), bins=20, alpha=0.3, label="Borda-Copeland")
    plt.xlabel("Final Rating")
    plt.ylabel("Number of Players")
    plt.title("Distribution of Final ELO Ratings")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/elo_final_distribution.png")
    plt.close()

    # Plot 