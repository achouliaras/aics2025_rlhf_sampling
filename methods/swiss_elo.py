import random
import numpy as np
from methods.utils import play_match_series, play_match, update_elo, update_elo_batch, rescale, evaluate_accuracy

# --- Simulation 1: swiss pairings ---
def simulate_swiss_tournament_elo(N, true_skills, initial_ratings, k=1, max_rounds=7, base_K=40, min_K=10, debug=False):
    """
    Swiss-system tournament simulation with Elo updates.
    
    Args:
        N: number of players
        true_skills: dict {player: true skill value}
        initial_ratings: dict {player: initial Elo ratings}
        k: number of games per match
        rounds: number of Swiss rounds
        base_K: starting K factor
        min_K: minimum K factor
    """
    players = list(initial_ratings.keys())
    ratings = initial_ratings.copy()
    swiss_scores = {p: 0 for p in players}   # still used for grouping
    games_played = {p: 0 for p in players}   # track games per player

    round_num = 0
    while True:
        if max_rounds is not None and round_num >= max_rounds:
            break

        # --- group players by Swiss score ---
        groups = {}
        for p in players:
            groups.setdefault(swiss_scores[p], []).append(p)

        new_pairs = []
        for group in groups.values():
            random.shuffle(group)
            # pair within group
            for i in range(0, len(group)-1, 2):
                new_pairs.append((group[i], group[i+1]))
            # handle odd leftover (bye = +1 point, no Elo change)
            if len(group) % 2 == 1:
                leftover = group[-1]
                swiss_scores[leftover] += 1

        # --- stopping condition: no more matches ---
        if not new_pairs:  
            break  

        # --- play matches for this round ---
        batch_updates = {p: {"opps": [], "scores": []} for p in players}

        for a, b in new_pairs:
            sa, sb = play_match_series(a, b, true_skills, k=k, mode="bt")
            swiss_scores[a] += sa
            swiss_scores[b] += sb
            games_played[a] += k
            games_played[b] += k

            # store results for batch Elo
            batch_updates[a]["opps"].append(ratings[b])
            batch_updates[a]["scores"].append(sa / k)  # normalize per game
            batch_updates[b]["opps"].append(ratings[a])
            batch_updates[b]["scores"].append(sb / k)

        # --- apply batch Elo updates ---
        for p in players:
            if batch_updates[p]["opps"]:  # skip players with no matches (bye)
                ratings[p] = update_elo_batch(
                    ratings[p],
                    batch_updates[p]["opps"],
                    batch_updates[p]["scores"],
                    games_played[p],
                    base_K=base_K*2,
                )
                swiss_scores[p] = ratings[p]

        round_num += 1  # increment round counter
    # Rescale final ratings to standard Elo range
    ratings = rescale(ratings, target_mean=1000, target_std=200)
    total_games = sum(games_played.values()) // 2
    if debug:
        print(f"Swiss tournament ended after {round_num} rounds and {total_games} games played.")
    return ratings, total_games


# --- Simulation 2: random paring + swiss pairings ---
def swiss_pairings(ratings):
    """Pair players by current Elo (sorted list, adjacent pairs)."""
    players_sorted = sorted(ratings, key=ratings.get, reverse=True)
    return [(players_sorted[i], players_sorted[i+1]) for i in range(0, len(players_sorted), 2)]


def simulate_rnd_swiss_elo(N, true_skills, initial_ratings, rnd_rounds = 1, rounds=10, K=40, debug=False):
    ratings = initial_ratings.copy()
    players = list(ratings.keys())
    games_played = 0
    # random matches
    init_K = K * 1  # higher K for initial rounds
    for rnd_round in range(rnd_rounds):
        players = list(ratings.keys())
        random.shuffle(players)
        for i in range(0, N-1, 2):
            p1, p2, s1, s2 = play_match(players[i], players[i+1], true_skills)
            ratings[p1] = update_elo(ratings[p1], ratings[p2], s1, init_K, rnd_round)
            ratings[p2] = update_elo(ratings[p2], ratings[p1], s2, init_K, rnd_round)
            games_played +=1
    # later rounds with swiss pairings
    for swiss_round in range(0, rounds):
        pairs = swiss_pairings(ratings)
        for a, b in pairs:
            assert a != b, "Self-match detected!"
            p1, p2, s1, s2 = play_match(a, b, true_skills)
            ratings[p1] = update_elo(ratings[p1], ratings[p2], s1, K, rnd_round+swiss_round)
            ratings[p2] = update_elo(ratings[p2], ratings[p1], s2, K, rnd_round+swiss_round)
            games_played +=1
    # Rescale final ratings to standard Elo range
    # ratings = rescale(ratings, target_mean=1000, target_std=200)
    if debug: print(f"Games played in random + swiss: {games_played}. Order accuracy: {evaluate_accuracy(sorted(ratings, key=lambda x: ratings[x], reverse=True), sorted(true_skills, key=lambda x: true_skills[x], reverse=True)):.3f}")
    return ratings


def simulate_rnd_swiss_elo_batch(N, true_skills, initial_ratings, rnd_rounds = 1, rounds=10, K=40, debug=False):
    ratings = initial_ratings.copy()
    players = list(ratings.keys())
    games_played = {p: 0 for p in players}   # track games per player
    # random matches
    init_K = K * 4  # higher K for initial rounds
    later_K = K * 1  # higher K for swiss rounds
    # --- initial random rounds ---
    for rnd_round in range(rnd_rounds):
        players = list(ratings.keys())
        random.shuffle(players)
        batch_updates = {p: {"opps": [], "scores": []} for p in players}

        for i in range(0, N-1, 2):
            p1, p2, s1, s2 = play_match(players[i], players[i+1], true_skills)
            games_played[p1] += 1
            games_played[p2] += 1

            # store results
            batch_updates[p1]["opps"].append(ratings[p2])
            batch_updates[p1]["scores"].append(s1)
            batch_updates[p2]["opps"].append(ratings[p1])
            batch_updates[p2]["scores"].append(s2)

        # apply batch updates
        for p in players:
            if batch_updates[p]["opps"]:
                ratings[p] = update_elo_batch(
                    ratings[p],
                    batch_updates[p]["opps"],
                    batch_updates[p]["scores"],
                    games_played[p],
                    base_K=init_K,
                )

    # --- later Swiss rounds ---
    for swiss_round in range(rounds):
        pairs = swiss_pairings(ratings)
        batch_updates = {p: {"opps": [], "scores": []} for p in players}

        for a, b in pairs:
            assert a != b, "Self-match detected!"
            p1, p2, s1, s2 = play_match(a, b, true_skills)
            games_played[p1] += 1
            games_played[p2] += 1

            batch_updates[p1]["opps"].append(ratings[p2])
            batch_updates[p1]["scores"].append(s1)
            batch_updates[p2]["opps"].append(ratings[p1])
            batch_updates[p2]["scores"].append(s2)

        # apply batch updates
        for p in players:
            if batch_updates[p]["opps"]:
                ratings[p] = update_elo_batch(
                    ratings[p],
                    batch_updates[p]["opps"],
                    batch_updates[p]["scores"],
                    games_played[p],
                    base_K=later_K
                )
    if debug: 
        total_games = sum(games_played.values()) // 2
        print("Games played in random + swiss: ", total_games)
    return ratings