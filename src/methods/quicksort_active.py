import numpy as np
import random
from methods.utils import play_match_series, evaluate_accuracy

def quicksort_active(order, true_skills, k=1, m=1, lo=0, hi=None, steps=0):
    """
    Recursive active Quicksort that:
      - stops if the array is already sorted
      - skips over already sorted prefixes
    order: list of player IDs
    true_skills: dict {player: skill rating}
    m: max number of partition steps
    lo, hi: recursion bounds
    steps: current step counter
    Returns: (sorted_order, games_played, steps_done)
    """
    if hi is None:
        hi = len(order) - 1

    # base case
    if lo >= hi or steps >= m:
        return order, 0, steps

    game_count = 0

    # --- early stopping: check if already sorted ---
    already_sorted = True
    for idx in range(lo, hi):
        s1, s2 = play_match_series(order[idx], order[idx+1], true_skills, k=k, mode="elo")
        game_count += 1*k
        if s1 == 0:  # means order[idx] lost → not sorted
            already_sorted = False
            break
    if already_sorted:
        return order, game_count, steps

    # --- choose pivot ---
    pivot_idx = random.randint(lo, hi)
    pivot = order[pivot_idx]

    # --- partition ---
    i, j = lo, hi
    while i <= j:
        while i <= hi:
            s1, s2 = play_match_series(order[i], pivot, true_skills, k=k, mode="elo")
            game_count += 1*k
            if s1 == 1:  # order[i] > pivot
                break
            i += 1
        while j >= lo:
            s1, s2 = play_match_series(order[j], pivot, true_skills, k=k, mode="elo")
            game_count += 1*k
            if s2 == 1:  # pivot > order[j]
                break
            j -= 1
        if i <= j:
            order[i], order[j] = order[j], order[i]
            i, j = i + 1, j - 1

    steps += 1

    # recurse on subarrays, skipping already sorted segments
    left_games, right_games = 0, 0
    if lo < j and steps < m:
        order, g, steps = quicksort_active(order, true_skills, k, m, lo, j, steps)
        left_games += g
    if i < hi and steps < m:
        order, g, steps = quicksort_active(order, true_skills, k, m, i, hi, steps)
        right_games += g

    return order, game_count + left_games + right_games, steps


def simulate_quicksort(N, true_skills, initial_ratings, k=25, rounds=1000, K=40, debug=False):
    """
    Simulate a Quicksort tournament and convert the final ranking to Elo-like ratings.
    """
    # Step 1: Simulate the Quicksort tournament
    order = list(true_skills.keys())
    games_played = 0
    for i in range(1):
        order, game_count, _ = quicksort_active(order, true_skills, k=k, m=rounds)
        games_played += game_count
    # Step 2: Convert the final order to Elo-like ratings
    ratings = {}
    for rank, player in enumerate(order):
        borda_score = N - rank
        ratings[player] = borda_score 
    # Rescale to Elo-like ratings
    score_values = np.array(list(ratings.values()))
    mu, sigma = np.mean(score_values), np.std(score_values)
    scaled_scores = 1000 + 200 * (score_values - mu) / sigma
    for i, player in enumerate(ratings.keys()):
        ratings[player] = scaled_scores[i]  
    if debug:
        true_order = sorted(order, key=lambda x: true_skills[x], reverse=True)
        print(f"Quicksort completed with {games_played} games. Order accuracy: {evaluate_accuracy(order, true_order):.3f}")
        # ground truth sorted order
    return ratings

