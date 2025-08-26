import random

def monty_hall_simulation(num_trials=10000, switch=True):
    wins = 0
    for _ in range(num_trials):
        doors = [0, 0, 1] 
        random.shuffle(doors)
        choice = random.randint(0, 2)
        # Host opens a goat door
        possible_doors = [i for i in range(3) if i != choice and doors[i] == 0]
        host_opens = random.choice(possible_doors)
        # Player switches or stays
        if switch:
            remaining = [i for i in range(3) if i != choice and i != host_opens][0]
            final_choice = remaining
        else:
            final_choice = choice
        if doors[final_choice] == 1:
            wins += 1
    return wins / num_trials

if __name__ == "__main__":
    trials = 10000
    win_rate_switch = monty_hall_simulation(trials, switch=True)
    win_rate_stay = monty_hall_simulation(trials, switch=False)
    print(f"Win rate when switching: {win_rate_switch:.2%}")
    print(f"Win rate when staying: {win_rate_stay:.2%}")