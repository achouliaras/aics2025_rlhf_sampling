import random
import numpy as np

# ---- ELO FUNCTIONS ----
def expected_score(rating_a, rating_b, mode="elo"):
    if mode == "elo":
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    elif mode == "bt":
        # Bradley-Terry rating calculation (simplified)
        return 1 / (1 + np.exp(rating_b - rating_a))
    else:
        raise ValueError("Unknown mode")

def play_match(a, b, true_skills, mode="elo", max_tie_prob=0.33, tau=None):
    """Simulate a match between players a and b based on true skills."""
    prob_p1_wins = expected_score(true_skills[a], true_skills[b], mode=mode)
    outcome = random.random()
    # auto-fit tau if not given
    if tau is None:
        skill_values = np.array(list(true_skills.values()))
        tau = np.std(skill_values) / 2  # adjustable factor
        tau = 200

    # tie probability decreases with skill gap
    skill_gap = abs(true_skills[a] - true_skills[b])
    tie_prob = max_tie_prob * np.exp(-skill_gap / tau)

    # rescale so win+loss+tie = 1
    prob_p1 = (1 - tie_prob) * prob_p1_wins
    prob_p2 = (1 - tie_prob) * (1 - prob_p1_wins)

    outcome = random.random()
    if outcome < prob_p1:
        return (a, b, 1, 0)
    elif outcome < prob_p1 + prob_p2:
        return (a, b, 0, 1)
    else:
        return (a, b, 0.5, 0.5)

def play_match_series(a, b, true_skills, k=2, mode="elo"):
    """Play k games between a and b, return match result (winner/loser/0.5 each if tied)."""
    wins_a, wins_b = 0, 0
    for _ in range(k):
        _, _, s1, s2 = play_match(a, b, true_skills, mode=mode)
        wins_a += s1
        wins_b += s2
        if wins_a > wins_b:
            return 1, 0
        elif wins_b > wins_a:
            return 0, 1
        else:
            return 0.5, 0.5
    # return wins_a, wins_b

def update_elo(rating_a, rating_b, score_a, base_K=40, games_played=0, mode="elo"):
    exp_a = expected_score(rating_a, rating_b, mode=mode)
    # --- K decay ---
    K=base_K
    # if games_played < 5:
    #     K = base_K
    # elif rating_a < 800 or rating_a > 1200:
    #     K = base_K / 4
    # elif rating_a < 600 or rating_a > 1400:
    #     K = base_K / 8
    # else:
    #     K = base_K / 2
    return rating_a + K * (score_a - exp_a)

def update_elo_batch(rating_a, opponents, scores_a, games_played, base_K=40, min_K=10, mode="elo"):
    """
    Batch Elo update for a single player against multiple opponents.

    Args:
        rating_a (float): Current rating of player A.
        opponents (list of float): Ratings of opponents.
        scores_a (list of float): Scores of player A vs each opponent (1=win, 0=loss, 0.5=draw).
        games_played (int): Total number of games A has played so far (for K decay).
        base_K (float): Initial K-factor.
        min_K (float): Lower bound for K (so learning doesn’t vanish).

    Returns:
        new_rating (float)
    """
    assert len(opponents) == len(scores_a), "Mismatch between opponents and scores"
    
    # --- K decay ---
    if games_played < 5:
        K = base_K
    elif rating_a < 800 or rating_a > 1200:
        K = base_K / 4
    elif rating_a < 600 or rating_a > 1400:
        K = base_K / 8
    else:
        K = base_K / 2

    # Expected score over all opponents
    exp_total = 0
    for r_b in opponents:
        exp_total += expected_score(rating_a, r_b, mode=mode)

    exp_avg = exp_total / len(opponents)
    score_avg = sum(scores_a) / len(scores_a)
    # Update Elo in batch
    new_rating = rating_a + K * (score_avg - exp_avg)
    return new_rating

def rescale(ratings, target_mean=1000, target_std=200):
    """Rescale ratings to look like Elo ratings."""
    values = np.array(list(ratings.values()))
    mean, std = values.mean(), values.std()
    scaled = {p: ((val - mean) / std) * target_std + target_mean for p, val in ratings.items()}
    return scaled

def evaluate_accuracy(order, true_order):
    """
    Compare the predicted order with the true skill ranking.
    Returns the accuracy as (# correct relative orderings) / (total possible pairs).
    """
    # compute pairwise accuracy
    correct, total = 0, 0
    n = len(order)
    pos_pred = {player: i for i, player in enumerate(order)}
    pos_true = {player: i for i, player in enumerate(true_order)}

    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = order[i], order[j]
            total += 1
            if (pos_pred[p1] < pos_pred[p2]) == (pos_true[p1] < pos_true[p2]):
                correct += 1

    return correct / total if total > 0 else 1.0