import itertools
from typing import List, Tuple, Dict, FrozenSet
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from tqdm import tqdm


def generate_coalitions(players: List[int]) -> List[FrozenSet[int]]:
    """
    Generate all possible coalitions (subsets) excluding the empty set.
    """
    coalitions = []
    for r in range(1, len(players) + 1):
        coalitions.extend([frozenset(c) for c in itertools.combinations(players, r)])
    return coalitions


def mean_variance_utility(alpha: float, mu: float, sigma2: float, eta: float) -> float:
    """
    Calculate the mean-variance utility for a player.
    """
    if (mu == 0) or (alpha == 0):
        return -10000000
    return (alpha * mu) ** (1 - eta) / (1 - eta) - (eta / 2) * (alpha**2) * sigma2 * (alpha * mu) ** (-1 - eta)


def calculate_utilities(
    coalition: FrozenSet[int],
    sharing_rule: Dict[int, float],
    mu: Dict[FrozenSet[int], Tuple[float, float]],
    eta: float,
) -> Dict[int, float]:
    """
    Calculate utilities for all players in a coalition based on the sharing rule.
    """
    if coalition not in mu:
        raise ValueError(f"Coalition {coalition} not found in characteristic function.")
    coalition_mu, coalition_sigma2 = mu[coalition]
    utilities = {}
    for player in coalition:
        alpha = sharing_rule.get(player, 0.0)
        utilities[player] = mean_variance_utility(alpha, coalition_mu, coalition_sigma2, eta)
    return utilities


def find_alpha_required(utility_grand: float, mu_pair: float, sigma2_pair: float, eta: float) -> float:
    """
    Solve for alpha' in the pair coalition that gives the target player at least
    their grand coalition utility.

    U = alpha' * mu_pair - (eta / 2) * (alpha')^2 * sigma2_pair >= utility_grand

    Returns the minimal alpha' that satisfies the inequality within [0,1].
    If no such alpha' exists, returns None.
    """
    a = -(eta / 2) * sigma2_pair
    b = mu_pair
    c = -utility_grand

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # No real solution

    sqrt_D = np.sqrt(discriminant)

    # Since a < 0, the quadratic opens downward.
    # We need alpha' >= the smaller root to satisfy U >= utility_grand
    alpha1 = (-b + sqrt_D) / (2 * a)
    alpha2 = (-b - sqrt_D) / (2 * a)

    # Sort roots
    alpha_min = min(alpha1, alpha2)
    alpha_max = max(alpha1, alpha2)

    # Feasible alpha' should be >= alpha_min
    if alpha_min >= 0 and alpha_min <= 1:
        return alpha_min
    elif alpha_max >= 0 and alpha_max <= 1:
        return alpha_max
    else:
        return None  # No feasible alpha' within [0,1]


def is_grand_coalition_stable(
    N: int,
    eta: float,
    mu: Dict[FrozenSet[int], Tuple[float, float]],
    grand_sharing_rule: Dict[int, float],
) -> bool:
    """
    Determine if the grand coalition is stable by checking possible deviations for all players.

    :param N: Number of players.
    :param eta: Risk aversion parameter.
    :param mu: Characteristic function mapping coalitions to (mean, variance).
    :param grand_sharing_rule: Sharing rule for the grand coalition.
    :return: True if stable, False otherwise.
    """
    players = list(range(1, N + 1))
    grand_coalition = frozenset(players)

    # Calculate utilities in the grand coalition
    grand_utils = calculate_utilities(grand_coalition, grand_sharing_rule, mu, eta)

    # Check Singleton Deviations
    for player in players:
        singleton_coalition = frozenset({player})
        singleton_mu, singleton_sigma2 = mu.get(singleton_coalition, (0.0, 0.0))
        alpha_singleton = 1.0  # Player takes the entire share
        singleton_utility = mean_variance_utility(alpha_singleton, singleton_mu, singleton_sigma2, eta)

        if singleton_utility > grand_utils[player]:
            # Player can achieve higher utility by deviating alone
            # print(f"Player {player} can deviate alone. Singleton utility: {singleton_utility} > Grand utility: {grand_utils[player]}")
            return False  # Grand coalition is not stable

    # Check Pair Deviations
    player_pairs = list(itertools.combinations(players, 2))
    for pair in player_pairs:
        pair_coalition = frozenset(pair)
        pair_mu, pair_sigma2 = mu.get(pair_coalition, (0.0, 0.0))
        player_i, player_j = pair

        utility_i_grand = grand_utils[player_i]
        utility_j_grand = grand_utils[player_j]

        # Find the minimal alpha_i' that gives player_i at least their grand utility
        alpha_i_required = find_alpha_required(utility_i_grand, pair_mu, pair_sigma2, eta)

        if alpha_i_required is None:
            # No feasible alpha_i' to satisfy player_i's utility
            continue  # Cannot deviate in this pair

        if alpha_i_required > 1.0:
            # Cannot assign more than 100% share
            continue  # Cannot deviate in this pair

        # Assign remaining share to player_j
        alpha_j = 1.0 - alpha_i_required

        # Calculate utilities in the pair coalition
        # utility_i = mean_variance_utility(alpha_i_required, pair_mu, pair_sigma2, eta)
        utility_j = mean_variance_utility(alpha_j, pair_mu, pair_sigma2, eta)

        if (utility_j > utility_j_grand) and not (np.isclose(utility_j, utility_j_grand, rtol=0.005)):
            # Both players can achieve at least their grand coalition utilities
            # print(f"Players {player_i} and {player_j} can deviate together.")
            # print(f"Player {player_i}: alpha={alpha_i_required}")
            # print(f"Player {player_j}: alpha={alpha_j}, utility={utility_j} >= {utility_j_grand}")
            return False  # Grand coalition is not stable

    # If no deviations are found, the grand coalition is stable
    return True


def generate_simplex_grid(step_size: float, N: int) -> List[Tuple[float, ...]]:
    """
    Generate a grid over the simplex for N players.
    Currently implemented for 3 players only.

    :param step_size: Step size for the grid.
    :param N: Number of players.
    :return: List of tuples representing shares for each player.
    """
    if N != 3:
        raise NotImplementedError("Simplex grid generation is implemented for 3 players only.")
    points = []
    n_steps = int(1 / step_size) + 1
    for i in range(n_steps + 1):
        alpha1 = i * step_size
        for j in range(n_steps + 1 - i):
            alpha2 = j * step_size
            alpha3 = 1.0 - alpha1 - alpha2
            if alpha3 < -1e-8 or alpha3 > 1 + 1e-8:
                continue
            alpha3 = max(alpha3, 0.0)
            points.append((alpha1, alpha2, alpha3))
    return points


def ternary_to_cartesian(alpha1: float, alpha2: float, alpha3: float) -> Tuple[float, float]:
    """
    Convert ternary coordinates to Cartesian coordinates for plotting.

    :param alpha1: Share of player 1.
    :param alpha2: Share of player 2.
    :param alpha3: Share of player 3.
    :return: Tuple representing (x, y) coordinates.
    """
    x = alpha2 + 0.5 * alpha3
    y = (np.sqrt(3) / 2) * alpha3
    return x, y


def plot_ternary_diagram(
    eta: float,
    mu: Dict[FrozenSet[int], Tuple[float, float]],
    stable_points: List[Tuple[float, float, float]],
    unstable_points: List[Tuple[float, float, float]],
):
    """
    Plot the stable and unstable points on a ternary diagram using matplotlib,
    and add interactivity to display (alpha1, alpha2, alpha3) on hover.
    Additionally, include a helper box displaying the characteristic function mu values.

    :param eta: Risk aversion parameter.
    :param mu: Characteristic function mapping coalitions to (mean, variance).
    :param stable_points: List of stable sharing rules (alpha1, alpha2, alpha3).
    :param unstable_points: List of unstable sharing rules (alpha1, alpha2, alpha3).
    """
    # Prepare data
    stable_points_cartesian = np.array([ternary_to_cartesian(*p) for p in stable_points])
    unstable_points_cartesian = np.array([ternary_to_cartesian(*p) for p in unstable_points])

    # Plotting
    plt.figure(figsize=(12, 10))

    # Plot the triangle boundary
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]
    plt.plot(triangle_x, triangle_y, "k-", linewidth=2)

    # Plot stable points
    if len(stable_points_cartesian) > 0:
        scatter_stable = plt.scatter(
            stable_points_cartesian[:, 0],
            stable_points_cartesian[:, 1],
            color="green",
            marker="o",
            label="Stable",
            s=50,
            alpha=0.7,
            edgecolors="w",
            linewidth=0.5,
        )

    # Plot unstable points
    if len(unstable_points_cartesian) > 0:
        scatter_unstable = plt.scatter(
            unstable_points_cartesian[:, 0],
            unstable_points_cartesian[:, 1],
            color="red",
            marker="x",
            label="Unstable",
            s=50,
            alpha=0.7,
            linewidth=1.0,
        )

    # Labels and title
    plt.title(f"Stable Sharing Rules in the Grand Coalition; η={eta}", fontsize=18, fontweight="bold")
    plt.legend(loc="upper right", fontsize=12)

    # Adjust plot limits and aspect
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, np.sqrt(3) / 2 + 0.05)
    plt.gca().set_aspect("equal", adjustable="box")

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    # Add labels to the corners
    offset = 0.03
    plt.text(-offset, -offset, "Alpha1", ha="center", va="center", fontsize=14, fontweight="bold")
    plt.text(1 + offset, -offset, "Alpha2", ha="center", va="center", fontsize=14, fontweight="bold")
    plt.text(0.5, np.sqrt(3) / 2 + offset, "Alpha3", ha="center", va="center", fontsize=14, fontweight="bold")

    # Remove the square box (axes frame)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Adding a helper box with mu values
    # Format the mu values for display
    mu_text_lines = []
    for coalition, (mean, variance) in mu.items():
        players = sorted(list(coalition))
        players_str = ",".join(map(str, players))
        mu_text_lines.append(f"{{{players_str}}}: μ={mean}, σ²={variance}")
    mu_text = "\n".join(mu_text_lines)

    # Position the helper box at the top-left corner
    plt.text(
        0.05,
        0.95,  # Relative coordinates
        f"Characteristic Function (μ):\n{mu_text}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Adding interactivity using mplcursors
    if len(stable_points_cartesian) > 0 and len(unstable_points_cartesian) > 0:
        cursor = mplcursors.cursor([scatter_stable, scatter_unstable], hover=True)
    elif len(stable_points_cartesian) > 0:
        cursor = mplcursors.cursor(scatter_stable, hover=True)
    elif len(unstable_points_cartesian) > 0:
        cursor = mplcursors.cursor(scatter_unstable, hover=True)
    else:
        cursor = None  # No points to annotate

    if cursor is not None:

        @cursor.connect("add")
        def on_add(sel):
            # Determine if the point is stable or unstable
            if sel.artist == scatter_stable:
                label = "Stable"
                if sel.index < len(stable_points):
                    alpha1, alpha2, alpha3 = stable_points[sel.index]
                else:
                    alpha1, alpha2, alpha3 = (0.0, 0.0, 0.0)
            else:
                label = "Unstable"
                if sel.index < len(unstable_points):
                    alpha1, alpha2, alpha3 = unstable_points[sel.index]
                else:
                    alpha1, alpha2, alpha3 = (0.0, 0.0, 0.0)
            # Set the annotation text
            sel.annotation.set(
                text=f"{label}\nAlpha1: {alpha1:.3f}\nAlpha2: {alpha2:.3f}\nAlpha3: {alpha3:.3f}",
                position=(0, 20),
                anncoords="offset points",
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    plt.show()


def main():
    N = 3
    eta = 10  # Risk aversion parameter (you can adjust this)
    grid_step_size = 0.005  # Adjust the step size for the grid (smaller step size for finer grid)

    # Define the characteristic function mu with (mean, variance) for each coalition
    mu = {
        frozenset({1}): (1.0, 0.0),  # Singleton coalitions have zero variance
        frozenset({2}): (3.0, 0.0),
        frozenset({3}): (5.0, 0.0),
        frozenset({1, 2}): (4.0, 1.0),
        frozenset({1, 3}): (6.0, 1.5),
        frozenset({2, 3}): (5.0, 2.0),
        frozenset({1, 2, 3}): (15.0, 3.0),
    }

    # Generate the simplex grid for the grand coalition sharing rules
    simplex_points = generate_simplex_grid(grid_step_size, N)

    # Prepare data for plotting
    stable_points = []
    unstable_points = []

    total_points = len(simplex_points)
    print(f"Total grand coalition sharing rules to evaluate: {total_points}")

    for alpha1, alpha2, alpha3 in tqdm(simplex_points, desc="Evaluating sharing rules"):
        # Define the sharing rule for the grand coalition
        grand_sharing_rule = {1: alpha1, 2: alpha2, 3: alpha3}

        # Check if the grand coalition is stable
        stable = is_grand_coalition_stable(N, eta, mu, grand_sharing_rule)

        # Collect the points
        if stable:
            stable_points.append((alpha1, alpha2, alpha3))
        else:
            unstable_points.append((alpha1, alpha2, alpha3))

    # Plot the results with interactivity
    plot_ternary_diagram(eta, mu, stable_points, unstable_points)


def single_check():
    eta = 1.0
    mu = {
        frozenset({1}): (1.0, 0.0),  # Singleton coalitions have zero variance
        frozenset({2}): (3.0, 0.0),
        frozenset({3}): (5.0, 0.0),
        frozenset({1, 2}): (5.0, 1.0),
        frozenset({1, 3}): (6.0, 1.5),
        frozenset({2, 3}): (5.0, 2.0),
        frozenset({1, 2, 3}): (15.0, 3.0),
    }
    grand_sharing_rule = {1: 0.11, 2: 0.21, 3: 0.68}
    is_grand_coalition_stable(3, eta, mu, grand_sharing_rule)


if __name__ == "__main__":
    main()
    # single_check()
