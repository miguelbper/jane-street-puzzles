from functools import cache


# functions
# ----------------------------------------------------------------------

def count(n: int) -> int:
    # Number of 1s in binary representation of n.
    bits = 0
    while n:
        bits += 1
        n &= n - 1
    return bits


def prob(seed: int, players: list[int]) -> float:
    ''' Given a list of players and (the index of) a seed, compute the 
    probability of seed being the winner of the tournament.
    
    '''

    @cache
    def prob_dp(state: int) -> float:
        ''' Probability of seed being the winner of the tournament, given 
        that we are in a current state. Here, 
        
            n = number of players
            s = number of subsets of players = 2**n
            state = a number in {0,...,s-1} representing a subset of 
                players (ith digit of state in bin = 1 iff i in subset)
        
        '''
        
        if not state & (1 << seed): # if seed not in state
            return 0
        if state == (1 << seed):    # if seed is the only player
            return 1

        m = count(state)      # num players left
        g = m // 2            # num games 
        t = 1 << g            # num outcomes
        curr_players = [k for k in range(n) if state & (1 << k)]

        ans = 0
        # compute ans using law of total probability
        # for each outcome, compute prob of that outcome happening
        #                   compute new_state if that outcome happens
        for i in range(t):     # loop over outcomes 
            
            new_prob = 1
            new_stat = state
            for j in range(g): # loop over games
                p1 = curr_players[2*j]            # player 1 of game j
                p2 = curr_players[2*j + 1]        # player 2 of game j
                winr = p2 if i & (1 << j) else p1 # winner of p1 vs p2
                losr = p1 if i & (1 << j) else p2 # loser  of p1 vs p2
                X = players[winr]                 # seed nr of winner
                Y = players[losr]                 # seed nr of loser
                new_prob *= Y / (X + Y)
                new_stat ^= 1 << losr

            ans += new_prob * prob_dp(new_stat)

        return ans

    n = len(players)
    s = 1 << n
    return prob_dp(s - 1)

    
# computations
# ----------------------------------------------------------------------

players = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# probability that each seed wins
probs = [prob(k, players) for k in range(16)]
for k in range(16):
    player2_str = ' <-' if players[k] == 2 else ''
    print(f'P({players[k]:2d} wins) = {probs[k]:.8f}{player2_str}')
print(f'       sum = {sum(probs):.8f}\n')

# probability that seed 2 wins for each swap
k0 = players.index(2)
p0 = probs[k0]

p_max = p0
i_max = 0
j_max = 0

for i in range(16):
    for j in range(i):
        swapped = list(players)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        
        k = swapped.index(2)
        p = prob(k, swapped)

        if p > p_max:
            p_max = p
            i_max = i
            j_max = j

        print(f'swap: seed_1 = {players[i]:2d} and seed_2 = {players[j]:2d} => P(2 wins) = {p:.8f}')

p_inc = p_max - p0
output = f'''\nbest swap:
    seed_1 = {players[i_max]} and seed_2 = {players[j_max]} 
    P( 2 wins) = {p0:.8f} (before swap)
    P( 2 wins) = {p_max:.8f} (after swap)
    P_increase = {p_inc:.8f}

'''
print(output)

'''
P( 1 wins) = 0.51921836
P(16 wins) = 0.00090356
P( 8 wins) = 0.00874121
P( 9 wins) = 0.00613649
P( 5 wins) = 0.03356150
P(12 wins) = 0.00266706
P( 4 wins) = 0.05668635
P(13 wins) = 0.00203873
P( 6 wins) = 0.02179862
P(11 wins) = 0.00367349
P( 3 wins) = 0.10678948
P(14 wins) = 0.00163814
P( 7 wins) = 0.01400661
P(10 wins) = 0.00484889
P( 2 wins) = 0.21603969 <-
P(15 wins) = 0.00125182
       sum = 1.00000000

swap: seed_1 = 16 and seed_2 =  1 => P(2 wins) = 0.21603969
swap: seed_1 =  8 and seed_2 =  1 => P(2 wins) = 0.22040336
swap: seed_1 =  8 and seed_2 = 16 => P(2 wins) = 0.22092167
swap: seed_1 =  9 and seed_2 =  1 => P(2 wins) = 0.22092167
swap: seed_1 =  9 and seed_2 = 16 => P(2 wins) = 0.22040336
swap: seed_1 =  9 and seed_2 =  8 => P(2 wins) = 0.21603969
swap: seed_1 =  5 and seed_2 =  1 => P(2 wins) = 0.22374757
swap: seed_1 =  5 and seed_2 = 16 => P(2 wins) = 0.22965225
swap: seed_1 =  5 and seed_2 =  8 => P(2 wins) = 0.21923192
swap: seed_1 =  5 and seed_2 =  9 => P(2 wins) = 0.21939472
swap: seed_1 = 12 and seed_2 =  1 => P(2 wins) = 0.23434027
swap: seed_1 = 12 and seed_2 = 16 => P(2 wins) = 0.21873229
swap: seed_1 = 12 and seed_2 =  8 => P(2 wins) = 0.21407437
swap: seed_1 = 12 and seed_2 =  9 => P(2 wins) = 0.21495919
swap: seed_1 = 12 and seed_2 =  5 => P(2 wins) = 0.21603969
swap: seed_1 =  4 and seed_2 =  1 => P(2 wins) = 0.22075101
swap: seed_1 =  4 and seed_2 = 16 => P(2 wins) = 0.23395869
swap: seed_1 =  4 and seed_2 =  8 => P(2 wins) = 0.22123251
swap: seed_1 =  4 and seed_2 =  9 => P(2 wins) = 0.22127467
swap: seed_1 =  4 and seed_2 =  5 => P(2 wins) = 0.21605116
swap: seed_1 =  4 and seed_2 = 12 => P(2 wins) = 0.21581740
swap: seed_1 = 13 and seed_2 =  1 => P(2 wins) = 0.23666489
swap: seed_1 = 13 and seed_2 = 16 => P(2 wins) = 0.21789796
swap: seed_1 = 13 and seed_2 =  8 => P(2 wins) = 0.21386448
swap: seed_1 = 13 and seed_2 =  9 => P(2 wins) = 0.21479096
swap: seed_1 = 13 and seed_2 =  5 => P(2 wins) = 0.21581740
swap: seed_1 = 13 and seed_2 = 12 => P(2 wins) = 0.21605116
swap: seed_1 = 13 and seed_2 =  4 => P(2 wins) = 0.21603969
swap: seed_1 =  6 and seed_2 =  1 => P(2 wins) = 0.21854894
swap: seed_1 =  6 and seed_2 = 16 => P(2 wins) = 0.23079318
swap: seed_1 =  6 and seed_2 =  8 => P(2 wins) = 0.22055841
swap: seed_1 =  6 and seed_2 =  9 => P(2 wins) = 0.22149382
swap: seed_1 =  6 and seed_2 =  5 => P(2 wins) = 0.21337205
swap: seed_1 =  6 and seed_2 = 12 => P(2 wins) = 0.21954745
swap: seed_1 =  6 and seed_2 =  4 => P(2 wins) = 0.20914019
swap: seed_1 =  6 and seed_2 = 13 => P(2 wins) = 0.21963123
swap: seed_1 = 11 and seed_2 =  1 => P(2 wins) = 0.22990122
swap: seed_1 = 11 and seed_2 = 16 => P(2 wins) = 0.21922322
swap: seed_1 = 11 and seed_2 =  8 => P(2 wins) = 0.21399856
swap: seed_1 = 11 and seed_2 =  9 => P(2 wins) = 0.21510420
swap: seed_1 = 11 and seed_2 =  5 => P(2 wins) = 0.21157325
swap: seed_1 = 11 and seed_2 = 12 => P(2 wins) = 0.21599413
swap: seed_1 = 11 and seed_2 =  4 => P(2 wins) = 0.20834784
swap: seed_1 = 11 and seed_2 = 13 => P(2 wins) = 0.21593659
swap: seed_1 = 11 and seed_2 =  6 => P(2 wins) = 0.21603969
swap: seed_1 =  3 and seed_2 =  1 => P(2 wins) = 0.19920466
swap: seed_1 =  3 and seed_2 = 16 => P(2 wins) = 0.28161919 <-
swap: seed_1 =  3 and seed_2 =  8 => P(2 wins) = 0.26032673
swap: seed_1 =  3 and seed_2 =  9 => P(2 wins) = 0.26232432
swap: seed_1 =  3 and seed_2 =  5 => P(2 wins) = 0.23814607
swap: seed_1 =  3 and seed_2 = 12 => P(2 wins) = 0.25554216
swap: seed_1 =  3 and seed_2 =  4 => P(2 wins) = 0.22913101
swap: seed_1 =  3 and seed_2 = 13 => P(2 wins) = 0.25615674
swap: seed_1 =  3 and seed_2 =  6 => P(2 wins) = 0.21720421
swap: seed_1 =  3 and seed_2 = 11 => P(2 wins) = 0.22120790
swap: seed_1 = 14 and seed_2 =  1 => P(2 wins) = 0.23969153
swap: seed_1 = 14 and seed_2 = 16 => P(2 wins) = 0.21638238
swap: seed_1 = 14 and seed_2 =  8 => P(2 wins) = 0.21612403
swap: seed_1 = 14 and seed_2 =  9 => P(2 wins) = 0.21682913
swap: seed_1 = 14 and seed_2 =  5 => P(2 wins) = 0.21638527
swap: seed_1 = 14 and seed_2 = 12 => P(2 wins) = 0.21696322
swap: seed_1 = 14 and seed_2 =  4 => P(2 wins) = 0.21390607
swap: seed_1 = 14 and seed_2 = 13 => P(2 wins) = 0.21648363
swap: seed_1 = 14 and seed_2 =  6 => P(2 wins) = 0.22120790
swap: seed_1 = 14 and seed_2 = 11 => P(2 wins) = 0.21720421
swap: seed_1 = 14 and seed_2 =  3 => P(2 wins) = 0.21603969
swap: seed_1 =  7 and seed_2 =  1 => P(2 wins) = 0.16877021
swap: seed_1 =  7 and seed_2 = 16 => P(2 wins) = 0.24086079
swap: seed_1 =  7 and seed_2 =  8 => P(2 wins) = 0.22085382
swap: seed_1 =  7 and seed_2 =  9 => P(2 wins) = 0.22431738
swap: seed_1 =  7 and seed_2 =  5 => P(2 wins) = 0.20328888
swap: seed_1 =  7 and seed_2 = 12 => P(2 wins) = 0.22815412
swap: seed_1 =  7 and seed_2 =  4 => P(2 wins) = 0.19323727
swap: seed_1 =  7 and seed_2 = 13 => P(2 wins) = 0.22925439
swap: seed_1 =  7 and seed_2 =  6 => P(2 wins) = 0.21224907
swap: seed_1 =  7 and seed_2 = 11 => P(2 wins) = 0.22579574
swap: seed_1 =  7 and seed_2 =  3 => P(2 wins) = 0.20491173
swap: seed_1 =  7 and seed_2 = 14 => P(2 wins) = 0.23272335
swap: seed_1 = 10 and seed_2 =  1 => P(2 wins) = 0.17450624
swap: seed_1 = 10 and seed_2 = 16 => P(2 wins) = 0.22391618
swap: seed_1 = 10 and seed_2 =  8 => P(2 wins) = 0.21131517
swap: seed_1 = 10 and seed_2 =  9 => P(2 wins) = 0.21411295
swap: seed_1 = 10 and seed_2 =  5 => P(2 wins) = 0.19921319
swap: seed_1 = 10 and seed_2 = 12 => P(2 wins) = 0.21785243
swap: seed_1 = 10 and seed_2 =  4 => P(2 wins) = 0.19077957
swap: seed_1 = 10 and seed_2 = 13 => P(2 wins) = 0.21842034
swap: seed_1 = 10 and seed_2 =  6 => P(2 wins) = 0.20865777
swap: seed_1 = 10 and seed_2 = 11 => P(2 wins) = 0.21708281
swap: seed_1 = 10 and seed_2 =  3 => P(2 wins) = 0.20918984
swap: seed_1 = 10 and seed_2 = 14 => P(2 wins) = 0.22062772
swap: seed_1 = 10 and seed_2 =  7 => P(2 wins) = 0.21603969
swap: seed_1 =  2 and seed_2 =  1 => P(2 wins) = 0.23028263
swap: seed_1 =  2 and seed_2 = 16 => P(2 wins) = 0.13780434
swap: seed_1 =  2 and seed_2 =  8 => P(2 wins) = 0.15281043
swap: seed_1 =  2 and seed_2 =  9 => P(2 wins) = 0.14963727
swap: seed_1 =  2 and seed_2 =  5 => P(2 wins) = 0.17526359
swap: seed_1 =  2 and seed_2 = 12 => P(2 wins) = 0.14916973
swap: seed_1 =  2 and seed_2 =  4 => P(2 wins) = 0.18404246
swap: seed_1 =  2 and seed_2 = 13 => P(2 wins) = 0.14751946
swap: seed_1 =  2 and seed_2 =  6 => P(2 wins) = 0.19434730
swap: seed_1 =  2 and seed_2 = 11 => P(2 wins) = 0.18011540
swap: seed_1 =  2 and seed_2 =  3 => P(2 wins) = 0.21074282
swap: seed_1 =  2 and seed_2 = 14 => P(2 wins) = 0.17517955
swap: seed_1 =  2 and seed_2 =  7 => P(2 wins) = 0.20668732
swap: seed_1 =  2 and seed_2 = 10 => P(2 wins) = 0.20287155
swap: seed_1 = 15 and seed_2 =  1 => P(2 wins) = 0.13582940
swap: seed_1 = 15 and seed_2 = 16 => P(2 wins) = 0.21821062
swap: seed_1 = 15 and seed_2 =  8 => P(2 wins) = 0.19325186
swap: seed_1 = 15 and seed_2 =  9 => P(2 wins) = 0.19866531
swap: seed_1 = 15 and seed_2 =  5 => P(2 wins) = 0.17394876
swap: seed_1 = 15 and seed_2 = 12 => P(2 wins) = 0.20981252
swap: seed_1 = 15 and seed_2 =  4 => P(2 wins) = 0.16285021
swap: seed_1 = 15 and seed_2 = 13 => P(2 wins) = 0.21215089
swap: seed_1 = 15 and seed_2 =  6 => P(2 wins) = 0.18609293
swap: seed_1 = 15 and seed_2 = 11 => P(2 wins) = 0.20676689
swap: seed_1 = 15 and seed_2 =  3 => P(2 wins) = 0.17581664
swap: seed_1 = 15 and seed_2 = 14 => P(2 wins) = 0.21379886
swap: seed_1 = 15 and seed_2 =  7 => P(2 wins) = 0.20287155
swap: seed_1 = 15 and seed_2 = 10 => P(2 wins) = 0.20668732
swap: seed_1 = 15 and seed_2 =  2 => P(2 wins) = 0.21603969

best swap:
    seed_1 = 3 and seed_2 = 16
    P( 2 wins) = 0.21603969 (before swap)
    P( 2 wins) = 0.28161919 (after swap)
    P_increase = 0.06557950
'''