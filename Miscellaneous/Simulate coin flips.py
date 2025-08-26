import random

def simulate_coin_flips(num_flips=100000):
    results = {'heads': 0, 'tails': 0}
    for _ in range(num_flips):
        flip = random.choice(['heads', 'tails'])
        results[flip] += 1
    print(f"Out of {num_flips} flips:")
    print(f"Heads: {results['heads']}")
    print(f"Tails: {results['tails']}")
    print(f"Probability of Tails: {results['tails']/num_flips}")
    print(f"Probability of Heads: {results['heads']/num_flips}")

simulate_coin_flips()