import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from methods.elo_random import simulate_elo_full
from methods.swiss_elo import simulate_rnd_swiss_elo, simulate_swiss_tournament_elo
from methods.borda_count import simulate_borda_count_rnd
from methods.bradley_terry_model import simulate_bt


# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    
    # --- Parameters ---
    N = 100
    K = 40
    initial_rating = 1000
    num_seeds = 100
    min_rounds = 5
    max_rounds = 200

    # --- Methods to compare ---
    methods = ["Elo RNG", "Elo RNG+Swiss", "Swiss Tournament", "Borda Wins", "Bradley Terry"]

    # --- Storage for results ---
    results = {m: {} for m in methods}

    # --- Run experiments ---
    for seed in range(num_seeds):
        rng = random.Random(seed)
        true_skills = {f"Player_{i+1}": rng.gauss(initial_rating, 200) for i in range(N)}
        ratings = {f"Player_{i+1}": initial_rating for i in range(N)}

        for rounds in range(min_rounds, max_rounds + 1):
            # Elo Full
            final_ratings_full, _, _ = simulate_elo_full(N, true_skills, ratings.copy(), rounds=rounds, K=K)
            corr_full, _ = spearmanr(
                list(true_skills.values()), list(final_ratings_full.values())
            )
            results["Elo RNG"].setdefault(rounds, []).append(corr_full)

            # Borda Count
            final_ratings_borda, _, _ = simulate_borda_count_rnd(N, true_skills, ratings.copy(), rounds=rounds)
            corr_borda, _ = spearmanr(
                list(true_skills.values()), list(final_ratings_borda.values())
            )
            results["Borda Wins"].setdefault(rounds, []).append(corr_borda)

            # Bradley-Terry
            final_ratings_bt, _, _ = simulate_bt(N, true_skills, ratings.copy(), games_per_player=rounds)
            corr_bt, _ = spearmanr(
                list(true_skills.values()), list(final_ratings_bt.values())
            )
            results["Bradley Terry"].setdefault(rounds, []).append(corr_bt)

            # Swiss Elo
            rnd_rounds = rounds // 2  # Half rounds RND, half Swiss
            swiss_rounds = rounds - rnd_rounds
            final_ratings_swiss, _, _ = simulate_rnd_swiss_elo(
                N, true_skills, ratings.copy(), rnd_rounds=rnd_rounds, rounds=swiss_rounds, K=K
            )
            corr_swiss, _ = spearmanr(
                list(true_skills.values()), list(final_ratings_swiss.values())
            )
            results["Elo RNG+Swiss"].setdefault(rounds, []).append(corr_swiss)

            # Swiss Tournament
            final_ratings_tournament, _, _ = simulate_swiss_tournament_elo(
                N, true_skills, ratings.copy(), k=1, rounds=rounds
            )
            corr_tournament, _ = spearmanr(
                list(true_skills.values()), list(final_ratings_tournament.values())
            )
            results["Swiss Tournament"].setdefault(rounds, []).append(corr_tournament)

            
        print(f"Seed {seed+1}, Rounds {rounds} done.")

    z = 1.96  # 95% CI
    round_values = range(min_rounds, max_rounds + 1)

    mean_corrs = {m: [] for m in methods}
    lower_ci = {m: [] for m in methods}
    upper_ci = {m: [] for m in methods}

    for m in methods:
        for r in round_values:
            vals = results[m][r]
            mu = np.mean(vals)
            sigma = np.std(vals, ddof=1)  # unbiased estimate
            ci = z * sigma / np.sqrt(len(vals))
            mean_corrs[m].append(mu)
            lower_ci[m].append(mu - ci)
            upper_ci[m].append(mu + ci)

    x = [i*N for i in round_values]  # Convert rounds to games

    # Save results to CSV
    df = pd.DataFrame({"Rounds": list(round_values)})
    for m in methods:
        df[f"{m}_mean"] = mean_corrs[m]
        df[f"{m}_lower_ci"] = lower_ci[m]
        df[f"{m}_upper_ci"] = upper_ci[m]
    df.to_csv("search_space_comparison.csv", index=False)

    
    # Plot with bands
    plt.figure(figsize=(10,6))
    for m in methods:
        plt.plot(x, mean_corrs[m], label=m)
        plt.fill_between(x, lower_ci[m], upper_ci[m], alpha=0.2)
    plt.xlabel("Number of Games Played")
    plt.ylabel("Spearman Correlation with True Skill")
    plt.title("Ranking Accuracy (100 seeds, 95% CI)")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/search_space_comparison.png")
