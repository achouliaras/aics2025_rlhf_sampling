import numpy as np
from methods.utils import play_match, evaluate_accuracy

def simulate_borda_round_robin(N, true_skills, initial_ratings, rounds=10, debug=False):
    players = list(initial_ratings.keys())
    scores = {p: 0 for p in players}
    games_played = 0

    for round_num in range(rounds):
        for i in range(N):
            for j in range(i+1, N):
                p1, p2 = players[i], players[j]
                _, _, s1, s2 = play_match(p1, p2, true_skills, mode="elo")
                scores[p1] += s1
                scores[p2] += s2
                games_played += 1
    # --- Scale scores to Elo scale (mean=1000, std=200) ---
    score_values = np.array(list(scores.values()))
    mu, sigma = np.mean(score_values), np.std(score_values)
    scaled_scores = 1000 + 200 * (score_values - mu) / sigma
    final_ratings = dict(zip(players, scaled_scores))
    if debug: print(f"Games played in round robin (Borda): {games_played}. Order accuracy: {evaluate_accuracy(sorted(final_ratings, key=lambda x: final_ratings[x], reverse=True), sorted(true_skills, key=lambda x: true_skills[x], reverse=True)):.3f}")
    return final_ratings
