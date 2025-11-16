import random
import numpy as np
from methods.utils import update_elo, play_match, evaluate_accuracy


# --- Simulation 1: full random rounds with history ---
def simulate_elo_full(N, true_skills, initial_ratings, rounds=40, K=40, debug=False):
    ratings = initial_ratings.copy()
    games_played = 0
    for round_num in range(rounds):
        players = list(ratings.keys())
        random.shuffle(players)  # still random outcomes
        for i in range(0, N-1, 2):
            p1, p2, s1, s2 = play_match(players[i], players[i+1], true_skills)
            ratings[p1] = update_elo(ratings[p1], ratings[p2], s1, K)
            ratings[p2] = update_elo(ratings[p2], ratings[p1], s2, K)
            games_played +=1
    if debug: 
        print(f"Games played in random matching: {games_played}. Order accuracy: {evaluate_accuracy(sorted(ratings, key=lambda x: ratings[x], reverse=True), sorted(true_skills, key=lambda x: true_skills[x], reverse=True)):.3f}")
    return ratings
