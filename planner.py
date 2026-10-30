
import numpy as np

# maximum distance an ambulance can cover
COVER_KM = 4.0


# calculate distances between areas
def distance_matrix(cent):
    xy = cent[["x_km", "y_km"]].to_numpy()

    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    return dist


# place ambulances one at a time
def place_ambulances(demand, dist, n, cover_km=COVER_KM):

    covers = dist <= cover_km
    remaining = np.asarray(demand, dtype=float).copy()

    picks = []

    for i in range(n):

        # find the area covering the most demand
        gain = covers @ remaining
        best = int(np.argmax(gain))

        picks.append(best)

        # reduce demand already covered
        remaining[covers[best]] *= 0.5

    return picks


# percentage of actual calls covered
def coverage(actual, dist, picks, cover_km=COVER_KM):

    actual = np.nan_to_num(np.asarray(actual, dtype=float))

    if actual.sum() == 0 or not picks:
        return float("nan")

    reached = (dist[picks] <= cover_km).any(axis=0)

    covered_calls = actual[reached].sum()
    total_calls = actual.sum()

    return covered_calls / total_calls


# average distance to nearest ambulance
def nearest_km(actual, dist, picks):

    actual = np.nan_to_num(np.asarray(actual, dtype=float))

    if actual.sum() == 0 or not picks:
        return float("nan")

    nearest = dist[picks].min(axis=0)

    total_distance = (nearest * actual).sum()
    total_calls = actual.sum()

    return total_distance / total_calls


# compare forecast with actual calls
def evaluate_year(fc, cent, method, n, cover_km=COVER_KM):

    dist = distance_matrix(cent)
    areas = cent["fsa"].tolist()

    results = []

    for block, group in fc.groupby("block"):

        group = group.set_index("fsa").reindex(areas)

        # skip missing data
        if group["calls"].isna().all():
            continue

        predicted = group[method].fillna(0)
        actual = group["calls"].fillna(0)

        # planned vs actual placement
        plan = place_ambulances(predicted, dist, n, cover_km)
        best = place_ambulances(actual, dist, n, cover_km)

        plan_coverage = coverage(actual, dist, plan, cover_km)
        best_coverage = coverage(actual, dist, best, cover_km)

        results.append((plan_coverage, best_coverage))

    results = np.array(results)

    return (
        np.nanmean(results[:, 0]),
        np.nanmean(results[:, 1])
    )


# optimize ambulance locations
def optimize_ambulances(demand, dist, n, cover_km=COVER_KM,
                        levels=3, backup_weight=0.001):

    from scipy.optimize import Bounds, LinearConstraint, milp

    demand = np.nan_to_num(np.asarray(demand, dtype=float))

    m = len(demand)

    # which areas can each ambulance reach
    covers = (dist <= cover_km).astype(float)

    # extra coverage is worth less
    weights = [1.0]

    for i in range(levels - 1):
        weights.append(backup_weight * 0.5 ** i)

    # decision variables
    total_vars = m + m * levels
    costs = np.zeros(total_vars)

    for i in range(levels):
        start = m + i * m
        end = start + m

        costs[start:end] = -weights[i] * demand

    # coverage constraints
    constraints = np.zeros((m, total_vars))
    constraints[:, :m] = -covers.T

    for i in range(levels):
        for j in range(m):
            constraints[j, m + i * m + j] = 1

    # number of ambulances
    total = np.zeros((1, total_vars))
    total[0, :m] = 1

    # variable limits
    upper = np.r_[
        np.full(m, n),
        np.ones(m * levels)
    ]

    # solve optimization problem
    result = milp(
        costs,
        constraints=[
            LinearConstraint(constraints, -np.inf, 0),
            LinearConstraint(total, n, n)
        ],
        integrality=np.ones(total_vars),
        bounds=Bounds(0, upper),
        options={"time_limit": 5}
    )

    # use simple placement if optimization fails
    if result.x is None:
        return place_ambulances(demand, dist, n, cover_km)

    locations = np.round(result.x[:m]).astype(int)

    picks = []

    for i in range(m):
        for j in range(locations[i]):
            picks.append(i)

    return picks
