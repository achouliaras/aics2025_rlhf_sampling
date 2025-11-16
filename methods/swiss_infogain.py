import random
import numpy as np
from methods.utils import expected_score, play_match_series, update_elo_batch, rescale, evaluate_accuracy

def pair_by_info_gain(players, ratings, discarded_pairs):
    """
    Pair players by maximizing information gain proxy.
    Avoids duplicate matches.
    """
    # Candidate edges (all pairs not played before)
    candidates = []
    for i in range(len(players)):
        for j in range(i+1, len(players)):
            a, b = players[i], players[j]
            if (a, b) in discarded_pairs or (b, a) in discarded_pairs:
                continue
            p_win = expected_score(ratings[a], ratings[b], mode="elo")
            ig = p_win * (1 - p_win)  # max info at ~0.5
            candidates.append((ig, a, b))

    # Sort by descending information gain
    candidates.sort(reverse=True, key=lambda x: x[0])
    # Select pairs greedily (Keep top 90% of pairs)
    top_n = int(len(candidates) * 1.0)
    candidates = candidates[:top_n]
    discarded_candidates = candidates[top_n:]
    for _, a, b in discarded_candidates:
        discarded_pairs.add((a, b))

    paired = set()
    pairs = []
    for _, a, b in candidates:
        if a not in paired and b not in paired:
            pairs.append((a, b))
            paired.add(a)
            paired.add(b)
    return pairs, discarded_pairs

def simulate_swiss_infogain(N, true_skills, initial_ratings, k=1, max_rounds=20, base_K=40, min_K=10, debug=False):
    """
    Swiss-style Active Learning Tournament with Information Gain pairing.
    """
    ratings = initial_ratings.copy()
    swiss_scores = {p: 0 for p in ratings}
    games_played = {p: 0 for p in ratings}
    players = list(ratings.keys())
    discarded_pairs = set()  # avoid repeats

    for rnd in range(max_rounds):
        # --- create info-gain pairs ---
        pairs, discarded_pairs = pair_by_info_gain(players, ratings, discarded_pairs)

        if not pairs:  # no more possible matches
            break

        # --- play matches for this round ---
        batch_updates = {p: {"opps": [], "scores": []} for p in players}
        for a, b in pairs:
            sa, sb = play_match_series(a, b, true_skills, k=k, mode="bt")
            swiss_scores[a] += sa
            swiss_scores[b] += sb
            games_played[a] += k
            games_played[b] += k

            # store results for batch Elo
            batch_updates[a]["opps"].append(ratings[b])
            batch_updates[a]["scores"].append(sa / k)
            batch_updates[b]["opps"].append(ratings[a])
            batch_updates[b]["scores"].append(sb / k)
        # print(f"Round {rnd+1}: {sum(games_played.values()) // 2} matches played.")
        # --- apply batch Elo updates ---
        for p in players:
            if batch_updates[p]["opps"]:  
                ratings[p] = update_elo_batch(
                    ratings[p],
                    batch_updates[p]["opps"],
                    batch_updates[p]["scores"],
                    games_played[p],
                    base_K=base_K,
                    min_K=min_K,
                )
                swiss_scores[p] = ratings[p]
    ratings = rescale(ratings, target_mean=1000, target_std=200)
    total_games = sum(games_played.values()) // 2
    if debug:
        order = sorted(ratings, key=lambda x: ratings[x], reverse=True)
        true_order = sorted(order, key=lambda x: true_skills[x], reverse=True)
        print(f"SwissInfoGain ended after {rnd+1} rounds and {total_games} games played. Order accuracy: {evaluate_accuracy(order, true_order)}")
    return ratings, total_games