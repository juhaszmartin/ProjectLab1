import itertools
from typing import List, Tuple, Dict, FrozenSet, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from scipy.optimize import brentq
from matplotlib.colors import LinearSegmentedColormap

float_type = np.longdouble


def generate_coalitions(players: List[int]) -> List[FrozenSet[int]]:
    coalitions = []
    for r in range(1, len(players) + 1):
        coalitions.extend([frozenset(c) for c in itertools.combinations(players, r)])
    return coalitions


def mean_variance_utility(alpha: float_type, mu: float_type, sigma2: float_type, eta: float_type) -> float_type:
    if (mu == 0) or (alpha == 0):
        return float_type(-1e7)
    term1 = (alpha * mu) ** (1 - eta) / (1 - eta)
    term2 = (eta / 2) * (alpha**2) * sigma2 * (alpha * mu) ** (-1 - eta)
    return term1 - term2


def calculate_utilities(
    coalition: FrozenSet[int],
    sharing_rule: Dict[int, float_type],
    mu: Dict[FrozenSet[int], Tuple[float_type, float_type]],
    eta: float_type,
) -> Dict[int, float_type]:
    if coalition not in mu:
        raise ValueError(f"Coalition {coalition} not found in characteristic function.")
    coalition_mu, coalition_sigma2 = mu[coalition]
    utilities = {}
    for player in coalition:
        alpha = sharing_rule.get(player, float_type(0.0))
        utilities[player] = mean_variance_utility(alpha, coalition_mu, coalition_sigma2, eta)
    return utilities


def find_alpha_required(utility_grand: float_type, mu_pair: float_type, sigma2_pair: float_type, eta: float_type) -> Optional[float_type]:
    """
    Solve for alpha' in the pair coalition that gives the target player at least
    their grand coalition utility using the complex utility function.
    Returns the minimal alpha' that satisfies the inequality within [0,1].
    If no such alpha' exists, returns None.
    """

    def utility_diff(alpha):
        return mean_variance_utility(alpha, mu_pair, sigma2_pair, eta) - utility_grand

    # Convert 0.0 and 1.0 to float_type
    zero = float_type(0.0)
    one = float_type(1.0)

    # Check if utility at alpha=1 is less than utility_grand
    if utility_diff(one) < 0:
        return None  # No feasible alpha' within [0,1]
    # Check if utility at alpha=0 is greater than utility_grand
    if utility_diff(zero) >= 0:
        return zero  # Minimum alpha' is 0

    # Find the root in the interval [0,1]
    try:
        alpha_required = brentq(utility_diff, zero, one, xtol=1e-14, rtol=1e-14, maxiter=10000)
        return float_type(alpha_required)
    except ValueError:
        return None  # No root found in [0,1]


def is_grand_coalition_stable(
    N: int,
    eta: float_type,
    mu: Dict[FrozenSet[int], Tuple[float_type, float_type]],
    grand_sharing_rule: Dict[int, float_type],
) -> bool:
    players = list(range(1, N + 1))
    grand_coalition = frozenset(players)
    grand_utils = calculate_utilities(grand_coalition, grand_sharing_rule, mu, eta)

    for player in players:
        singleton_coalition = frozenset({player})
        singleton_mu, singleton_sigma2 = mu.get(singleton_coalition, (float_type(0.0), float_type(0.0)))
        alpha_singleton = float_type(1.0)
        singleton_utility = mean_variance_utility(alpha_singleton, singleton_mu, singleton_sigma2, eta)
        if singleton_utility > grand_utils[player]:
            return False

    player_pairs = list(itertools.combinations(players, 2))
    for pair in player_pairs:
        pair_coalition = frozenset(pair)
        pair_mu, pair_sigma2 = mu.get(pair_coalition, (float_type(0.0), float_type(0.0)))
        player_i, player_j = pair

        utility_i_grand = grand_utils[player_i]
        utility_j_grand = grand_utils[player_j]

        alpha_i_required = find_alpha_required(utility_i_grand, pair_mu, pair_sigma2, eta)
        if alpha_i_required is None or alpha_i_required > 1.0:
            continue

        alpha_j = float_type(1.0) - alpha_i_required
        utility_j = mean_variance_utility(alpha_j, pair_mu, pair_sigma2, eta)
        epsilon = float_type(1e-11)
        if utility_j > utility_j_grand + epsilon:
            return False

    return True


def generate_simplex_grid(step_size: float_type, N: int) -> List[Tuple[float_type, ...]]:
    if N != 3:
        raise NotImplementedError("Simplex grid generation is implemented for 3 players only.")
    points = []
    n_steps = int(1 / step_size)
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            alpha1 = float_type(i) / n_steps
            alpha2 = float_type(j) / n_steps
            alpha3 = float_type(k) / n_steps
            points.append((alpha1, alpha2, alpha3))
    return points


def ternary_to_cartesian(alpha1: float_type, alpha2: float_type, alpha3: float_type) -> Tuple[float_type, float_type]:
    x = alpha2 + float_type(0.5) * alpha3
    y = (np.sqrt(3) / 2) * alpha3
    return x, y


def format_mu(mu: Dict[FrozenSet[int], Tuple[float_type, float_type]]) -> str:
    """
    Format the characteristic function mu as a string for display.
    """
    mu_text_lines = []
    for coalition, (mean, variance) in mu.items():
        players = sorted(list(coalition))
        players_str = ",".join(map(str, players))
        mu_text_lines.append(f"{{{players_str}}}: μ={float(mean):.1f}, σ²={float(variance):.1f}")
    return "\n".join(mu_text_lines)


def plot_comparison(eta_list, mu_set_1, mu_set_2, grid_step_size, N):
    simplex_points = generate_simplex_grid(grid_step_size, N)

    # Define the custom discrete colormap
    num_eta = len(eta_list)
    num_colors = num_eta + 1  # Possible counts: 0,1,2,...,num_eta
    jet_cmap = plt.cm.jet
    custom_colors = [(0, 0, 0)] + [jet_cmap(i) for i in np.linspace(0, 1, num_colors - 1)]
    custom_cmap = LinearSegmentedColormap.from_list("custom_jet_black", custom_colors, N=num_colors)

    boundaries = np.arange(num_colors + 1) - 0.5  # Boundaries between integers
    norm = BoundaryNorm(boundaries, ncolors=num_colors, clip=True)

    stability_count_1 = {point: 0 for point in simplex_points}
    stability_count_2 = {point: 0 for point in simplex_points}

    # Evaluate stability for each eta and each point in both mu sets
    for eta in eta_list:
        eta = float_type(eta)
        for point in tqdm(simplex_points, desc=f"Evaluating for eta={eta}"):
            grand_sharing_rule = {1: point[0], 2: point[1], 3: point[2]}

            if is_grand_coalition_stable(N, eta, mu_set_1, grand_sharing_rule):
                stability_count_1[point] += 1
            if is_grand_coalition_stable(N, eta, mu_set_2, grand_sharing_rule):
                stability_count_2[point] += 1

    # Prepare data for plotting
    points_cartesian_1 = [ternary_to_cartesian(*p) for p in simplex_points]
    points_cartesian_2 = [ternary_to_cartesian(*p) for p in simplex_points]

    stability_values_1 = [stability_count_1[p] for p in simplex_points]
    stability_values_2 = [stability_count_2[p] for p in simplex_points]

    # Plotting with GridSpec
    fig = plt.figure(figsize=(17, 9))
    gs = GridSpec(1, 3, width_ratios=[2, 2, 0.2], wspace=0.3)  # Adjust wspace as needed

    # Add a joint title centered above both plots
    eta_str = ", ".join(map(str, eta_list))
    fig.suptitle(f"Stable Sharing Rules in the Grand Coalition; η={eta_str}", fontsize=18, fontweight="bold")

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Scatter plot for Set 1
    scatter1 = ax1.scatter(
        [float(p[0]) for p in points_cartesian_1],
        [float(p[1]) for p in points_cartesian_1],
        c=stability_values_1,
        cmap=custom_cmap,
        norm=norm,
        edgecolor="k",
        s=9,
        linewidths=0.01,
    )
    ax1.set_title("Stability Comparison (Set 1)", fontsize=14, pad=20, fontweight="bold")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Add corner labels for Set 1
    offset = 0.03
    ax1.text(-offset, -offset, "Alpha1", ha="center", va="center", fontsize=14, fontweight="bold")
    ax1.text(1 + offset, -offset, "Alpha2", ha="center", va="center", fontsize=14, fontweight="bold")
    ax1.text(0.5, np.sqrt(3) / 2 + offset, "Alpha3", ha="center", va="center", fontsize=14, fontweight="bold")

    # Add helper box for Set 1
    mu_text1 = format_mu(mu_set_1)
    fig.text(
        0.03,  # x-position (left side)
        0.68,
        f"Payoff X:\n{mu_text1}",
        fontsize=9.5,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Scatter plot for Set 2
    scatter2 = ax2.scatter(
        [float(p[0]) for p in points_cartesian_2],
        [float(p[1]) for p in points_cartesian_2],
        c=stability_values_2,
        cmap=custom_cmap,
        norm=norm,
        edgecolor="k",
        s=9,
        linewidths=0.01,
    )
    ax2.set_title("Stability Comparison (Set 2)", fontsize=14, pad=20, fontweight="bold")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add corner labels for Set 2
    ax2.text(-offset, -offset, "Alpha1", ha="center", va="center", fontsize=14, fontweight="bold")
    ax2.text(1 + offset, -offset, "Alpha2", ha="center", va="center", fontsize=14, fontweight="bold")
    ax2.text(0.5, np.sqrt(3) / 2 + offset, "Alpha3", ha="center", va="center", fontsize=14, fontweight="bold")

    # Add helper box for Set 2
    mu_text2 = format_mu(mu_set_2)
    fig.text(
        0.45,  # x-position (right side)
        0.68,
        f"Payoff X:\n{mu_text2}",
        fontsize=9.5,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Add colorbar in the third GridSpec column
    cbar_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(scatter2, cax=cbar_ax, boundaries=boundaries, ticks=np.arange(num_colors))
    cbar.set_label("Number of Stable Etas", fontsize=15)
    cbar.set_ticks(np.arange(num_colors))
    cbar.set_ticklabels([str(i) for i in range(num_colors)])
    cbar.ax.tick_params(labelsize=10)

    # Adjust layout to make room for the suptitle and helper boxes
    plt.tight_layout(rect=[0.020, 0.03, 0.965, 0.95])  # [left, bottom, right, top]
    plt.show()


if __name__ == "__main__":
    eta_list = [0, 2.5, 5, 7.5, 10]  # List of etas to evaluate
    grid_step_size = float_type(0.01)  # Step size for the grid

    # Define two sets of mu values, converting to float_type
    mu_set_1 = {
        frozenset({1}): (float_type(4.0), float_type(4.0)),
        frozenset({2}): (float_type(4.0), float_type(4.0)),
        frozenset({3}): (float_type(4.0), float_type(4.0)),
        frozenset({1, 2}): (float_type(10.0), float_type(7.0)),
        frozenset({1, 3}): (float_type(10.0), float_type(7.0)),
        frozenset({2, 3}): (float_type(10.0), float_type(7.0)),
        frozenset({1, 2, 3}): (float_type(20.0), float_type(2.0)),
    }

    mu_set_2 = {
        frozenset({1}): (float_type(4.0), float_type(4.0)),
        frozenset({2}): (float_type(4.0), float_type(4.0)),
        frozenset({3}): (float_type(4.0), float_type(4.0)),
        frozenset({1, 2}): (float_type(10.0), float_type(7.0)),
        frozenset({1, 3}): (float_type(10.0), float_type(7.0)),
        frozenset({2, 3}): (float_type(10.0), float_type(7.0)),
        frozenset({1, 2, 3}): (float_type(20.0), float_type(75.0)),
    }

    plot_comparison(eta_list, mu_set_1, mu_set_2, grid_step_size, 3)
