import random
import numpy as np
import pandas as pd
from methods.elo_roundrobin import simulate_elo_roundrobin
from methods.borda_count_round_robin import simulate_borda_round_robin
from methods.swiss_infogain import simulate_swiss_infogain
import warnings

def run_experiment_swiss_infogain(N, initial_rating, rounds_list, num_seeds=100, K=40):
    results = []

    for rounds in rounds_list:
        total_games = []
        for seed in range(num_seeds):
            random.seed(seed)
            np.random.seed(seed)

            # --- init players ---
            ratings0 = {f"Player_{i+1}": initial_rating for i in range(N)}
            true_skills = {f"Player_{i+1}": random.gauss(initial_rating, 200) for i in range(N)}
            
            # --- simulate ---
            final_ratings, games = simulate_swiss_infogain(
                N, true_skills, ratings0.copy(), max_rounds=rounds, base_K=K, debug=False
            )
            total_games.append(games)
            df = pd.DataFrame({
                "Player": list(true_skills.keys()),
                "TrueSkill": list(true_skills.values()),
                "SwissInfoGain": list(final_ratings.values())
            })
            corr_infogain = df["TrueSkill"].corr(df["SwissInfoGain"])
            results.append({
                "Rounds": rounds,
                "Seed": seed,
                "SwissInfoGain": corr_infogain
            })
        avg_games = np.mean(total_games)
        results[-1]["AvgGames"] = avg_games  # store average games for last seed of this round

    df=pd.DataFrame(results)
    agg = df.groupby("Rounds").agg(["mean", "std"])
    agg.columns = ["_".join(col) for col in agg.columns]
    agg = agg.reset_index()

    # 95% CI ~ mean ± 1.96 * std / sqrt(n)
    n = df["Seed"].nunique()
    for method in ["SwissInfoGain"]:
        agg[f"{method}_lower_ci"] = agg[f"{method}_mean"] - 1.96 * agg[f"{method}_std"] / np.sqrt(n)
        agg[f"{method}_upper_ci"] = agg[f"{method}_mean"] + 1.96 * agg[f"{method}_std"] / np.sqrt(n)
    return agg

def run_experiment_rr(N, initial_rating, rounds_list, num_seeds=100, K=40):
    results = []

    for rounds in rounds_list:
        for seed in range(num_seeds):
            random.seed(seed)
            np.random.seed(seed)

            # --- init players ---
            ratings0 = {f"Player_{i+1}": initial_rating for i in range(N)}
            true_skills = {f"Player_{i+1}": random.gauss(initial_rating, 200) for i in range(N)}

            # --- simulate ---
            final_ratings_rr = simulate_elo_roundrobin(
                N, true_skills, ratings0.copy(), rounds=rounds, K=K, debug=False
            )
            final_ratings_borda = simulate_borda_round_robin(
                N, true_skills, ratings0.copy(), rounds=rounds, debug=False
            )
            df = pd.DataFrame({
                "Player": list(true_skills.keys()),
                "TrueSkill": list(true_skills.values()),
                "Elo-RR": list(final_ratings_rr.values()),
                "Borda-RR": list(final_ratings_borda.values()),
            })
            corr_rr = df["TrueSkill"].corr(df["Elo-RR"])
            corr_borda = df["TrueSkill"].corr(df["Borda-RR"])
            results.append({
                "Rounds": rounds,
                "Seed": seed,
                "Elo-RR": corr_rr,
                "Borda-RR": corr_borda
            })
    
    df=pd.DataFrame(results)
    agg = df.groupby("Rounds").agg(["mean", "std"])
    agg.columns = ["_".join(col) for col in agg.columns]
    agg = agg.reset_index()

    # 95% CI ~ mean ± 1.96 * std / sqrt(n)
    n = df["Seed"].nunique()
    for method in ["Elo-RR", "Borda-RR"]:
        agg[f"{method}_lower_ci"] = agg[f"{method}_mean"] - 1.96 * agg[f"{method}_std"] / np.sqrt(n)
        agg[f"{method}_upper_ci"] = agg[f"{method}_mean"] + 1.96 * agg[f"{method}_std"] / np.sqrt(n)
    return agg

if __name__ == "__main__":
    # --- Parameters ---
    N = 100
    K = 40
    initial_rating = 1000
    num_seeds = 100
    min_rounds = 5
    max_rounds = 60
    
    # # --- Run Round Robin/Borda as before ---
    # rounds_list = [1, 2, 3, 4]  
    # df_agg = run_experiment_rr(N, initial_rating, rounds_list, num_seeds=num_seeds)
    # # save to CSV
    # df_agg.to_csv("round_robin_results_agg.csv", index=False)

    # --- Run Swiss InfoGain ---
    # suppress warnings
    warnings.filterwarnings('ignore')
    rounds_list = list(range(min_rounds, max_rounds + 1, 5))
    df_agg_swiss = run_experiment_swiss_infogain(N, initial_rating, rounds_list, num_seeds=num_seeds)
    # save to CSV
    df_agg_swiss.to_csv("swiss_infogain_results_agg.csv", index=False)