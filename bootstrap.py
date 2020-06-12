import numpy
import random


def bootstrapped_sample(observations):
    '''
    :observations ~ [tuple, ...]
    :observations ~ [(1,), (0,) ...]
    :observations ~ [(1, 0, 0), (0, 1, 0), ...]
    '''
    index_min = 0
    index_max = len(observations) - 1
    resampled_observations = []
    for i in range(len(observations)):
        resampled_observations.append(observations[random.randint(index_min, index_max)])
    return resampled_observations


def bootstrapped_average(sample, n_resamples=1000, alpha=5):
    avgs = []
    for _ in range(n_resamples):
        bss = bootstrapped_sample(sample)
        avg = sum(bss) / len(bss)
        avgs.append(avg)

    ntiles = [alpha, 50, 100 - alpha]
    return numpy.percentile(avgs, ntiles)


def adjusted_win_rate_test(wins, losses, draws, num_bootstraps):
    observations = []
    observations.extend([(1, 0, 0)] * wins)
    observations.extend([(0, 1, 0)] * losses)
    observations.extend([(0, 0, 1)] * draws)

    def record_from_observations(obs):
        wins = sum(x[0] for x in obs)
        losses = sum(x[1] for x in obs)
        draws = sum(x[2] for x in obs)
        return (wins, losses, draws)

    def adjusted_win_rate(obs):
        wins = sum(x[0] for x in obs)
        losses = sum(x[1] for x in obs)
        draws = sum(x[2] for x in obs)
        total = wins + losses + draws
        return (wins / total) + ((draws / total) * 0.5)

    adjusted_win_rates = []
    for i in range(num_bootstraps):
        bs = bootstrapped_sample(observations)
        adjusted_win_rates.append(adjusted_win_rate(bs))

    print("\nOriginal")
    print("\nRecord:", record_from_observations(observations))
    print("AWR:", adjusted_win_rate(observations))

    print("\n\nBootstrapped")
    ntiles = [1, 5, 10, 50, 90, 95, 99]
    p_ntiles = numpy.percentile(adjusted_win_rates, ntiles)
    print("\n{col1:<30}{col2:<30}".format(col1="NTILE", col2="AWR"))
    for ntile, pntile in zip(ntiles, p_ntiles):
        print("{ntile:<30}{pntile:<30}".format(ntile=ntile, pntile=round(pntile, 2)))
    print()


def games_to_play_analysis():
    wins = 40
    losses = 40
    draws = 20
    for i in (.2, .5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0):
        wp = int(wins * i)
        lp = int(losses * i)
        dp = int(draws * i)
        print("\n\nSCENARIO:", wp, lp, dp, wp + lp + dp)
        adjusted_win_rate_test(wp, lp, dp, 1000)


if __name__ == "__main__":
    import sys
    wins, losses, draws = sys.argv[1].strip().split("-")
    wins = int(wins)
    losses = int(losses)
    draws = int(draws)
    adjusted_win_rate_test(wins, losses, draws, 1000)
    # games_to_play_analysis()
