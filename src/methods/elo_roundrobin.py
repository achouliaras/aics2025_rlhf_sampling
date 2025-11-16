from methods.utils import play_match, update_elo

def simulate_elo_roundrobin(N, true_skills, initial_ratings, rounds=10, K=40, debug=False):
    ratings = initial_ratings.copy()
    players = list(ratings.keys())
    games_played = 0
    for round_num in range(rounds):
        for i in range(N):
            for j in range(i+1, N):
                p1, p2 = players[i], players[j]
                _, _, s1, s2 = play_match(p1, p2, true_skills)
                # Update Elo for both players
                new_r1 = update_elo(ratings[p1], ratings[p2], s1, K)
                new_r2 = update_elo(ratings[p2], ratings[p1], s2, K)
                ratings[p1], ratings[p2] = new_r1, new_r2
                games_played += 1
    if debug: print("Games played in round robin (Elo):", games_played)
    return ratings
