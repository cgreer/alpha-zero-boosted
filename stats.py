import numpy

from bootstrap import bootstrapped_average


def align_neg(x):
    x = str(x)
    if x[0] != "-":
        x = " " + x
    return x


def describe_sample(samples):
    ntiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    ptiles = numpy.percentile(samples, ntiles)

    s_count = len(samples)

    s_avg = sum(samples) / len(samples)
    s_avg = round(s_avg, 4)

    s_avg_low, _, s_avg_high = bootstrapped_average(samples)
    s_avg_low = round(s_avg_low, 4)
    s_avg_high = round(s_avg_high, 4)

    s_min = min(samples)
    s_min = round(s_min, 4)

    s_max = max(samples)
    s_max = round(s_max, 4)

    print(f"{'count':<15}{align_neg(s_count):<15}")
    print(f"{'avg (@5)':<15}{align_neg(s_avg_low):<15}")
    print(f"{'avg':<15}{align_neg(s_avg):<15}")
    print(f"{'avg (@95)':<15}{align_neg(s_avg_high):<15}")
    print(f"{'min':<15}{align_neg(s_min):<15}")
    print(f"{'max':<15}{align_neg(s_max):<15}")
    for i, ntile in enumerate(ntiles):
        ptile = str(round(ptiles[i], 4))
        ptile = align_neg(ptile)
        ntile = "@" + str(ntile)
        print(f"{ntile:<15}{ptile:<15}")


x = numpy.random.normal(0, 1, 1000)
describe_sample(x)
