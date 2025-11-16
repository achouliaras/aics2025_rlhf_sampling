import random
import numpy as np
from methods.utils import play_match, evaluate_accuracy

# --- Borda Count: find rating based on the number of wins ---
def simulate_borda_count_rnd(N, true_skills, initial_ratings, rounds=40, debug=False):
    ratings = initial_ratings.copy()
    players = list(ratings.keys())
    games_played = 0
    wins = {p: 0 for p in players}
    for round_num in range(rounds):
        random.shuffle(players)  # still random outcomes
        for i in range(0, N-1, 2):
            p1, p2, s1, s2 = play_match(players[i], players[i+1], true_skills, mode="elo")
            wins[p1] += s1
            wins[p2] += s2
            games_played += 1
    # --- Scale scores to Elo scale (mean=1000, std=200) ---
    score_values = np.array(list(wins.values()))
    mu, sigma = np.mean(score_values), np.std(score_values)
    scaled_scores = 1000 + 200 * (score_values - mu) / sigma
    final_ratings = dict(zip(players, scaled_scores))
    if debug: print(f"Games played in random matching (Borda): {games_played}. Order accuracy: {evaluate_accuracy(sorted(final_ratings, key=lambda x: final_ratings[x], reverse=True), sorted(true_skills, key=lambda x: true_skills[x], reverse=True)):.3f}")
    return final_ratings