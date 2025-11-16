import random
import numpy as np
from methods.utils import play_match, evaluate_accuracy

# ---- BRADLEY–TERRY ONE-SHOT SIMULATION ----
def bt_expected_score(rating_a, rating_b):
    """Probability that A beats B under Bradley–Terry."""
    return np.exp(rating_a) / (np.exp(rating_a) + np.exp(rating_b))

def fit_bt(skills, matches, iters=50, lr=0.01):
    """
    Fit Bradley–Terry model by gradient ascent.
    skills: dict {player: skill_param}
    matches: list of (a, b, s1, s2) outcomes
    """
    skill_vec = skills.copy()
    for _ in range(iters):
        grad = {p: 0.0 for p in skill_vec}
        for a, b, s1, s2 in matches:
            pa = bt_expected_score(skill_vec[a], skill_vec[b])
            # Treat draws as half-wins
            grad[a] += (s1 - pa)
            grad[b] += (s2 - (1 - pa))
        # Update
        for p in skill_vec:
            skill_vec[p] += lr * grad[p]
    return skill_vec

def rescale_bt(bt_ratings, target_mean=1000, target_std=200):
    """Rescale BT log-odds to look like Elo ratings."""
    values = np.array(list(bt_ratings.values()))
    mean, std = values.mean(), values.std()
    scaled = {p: ((val - mean) / std) * target_std + target_mean for p, val in bt_ratings.items()}
    return scaled

def simulate_bt(N, true_skills, initial_ratings, games_per_player=20, debug=False):
    ratings = initial_ratings.copy()
    players = list(initial_ratings.keys())
    matches = []
    games_played = 0
    for _ in range(games_per_player * N // 2):
        p1, p2 = random.sample(players, 2)
        a, b, s1, s2 = play_match(p1, p2, true_skills, mode="bt")
        matches.append((a, b, s1, s2))
        games_played += 1
    # Initial skills = 0 for all
    init_skill_guess = {p: 0.0 for p in players}
    bt_skill_guess = fit_bt(init_skill_guess, matches, iters=20, lr=0.01)
    ratings = rescale_bt(bt_skill_guess, target_mean=1000, target_std=200)
    if debug: print(f"Games played with Bradley-Terry: {games_played}. Order accuracy: {evaluate_accuracy(sorted(ratings, key=lambda x: ratings[x], reverse=True), sorted(true_skills, key=lambda x: true_skills[x], reverse=True)):.3f}")
    return ratings
