import itertools
from typing import List, Tuple, Dict, FrozenSet
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from tqdm import tqdm

# NOTE: just check e.g. what player1 needs in {1,2} at least, and then check if player2 is happ with the rest
# --> less computation


def generate_coalitions(players: List[int]) -> List[FrozenSet[int]]:
    """
    Generate all possible coalitions (subsets) excluding the empty set.
    """
    coalitions = []
    for r in range(1, len(players) + 1):
        coalitions.extend([frozenset(c) for c in itertools.combinations(players, r)])
    return coalitions


def expected_utility(
    player: int,
    coalition: FrozenSet[int],
    mu_S: List[Tuple[float, float]],
    sharing_rule: Dict[int, float],
    eta: float,
) -> float:
    """
    Calculate the expected utility for a player in a coalition.
    """
    exp_util = 0.0
    for payoff, prob in mu_S:
        # Allocate share based on sharing rule
        share = sharing_rule.get(player, 0) * payoff
        # Compute utility: handle share <= 0
        if share > 0:
            if eta != 1:
                utility = (share ** (1 - eta) - 1) / (1 - eta)
            else:
                utility = np.log(share)
        elif share == 0:
            if eta < 1:
                utility = (-1) / (1 - eta)
            else:
                utility = -1e10  # Assign a large negative utility
        else:
            # share < 0, which should not happen
            raise ValueError(f"Negative share encountered: {share}, {payoff}, {sharing_rule.get(player)}")
        exp_util += prob * utility
    return exp_util


def calculate_all_expected_utilities(
    N: int,
    eta: float,
    mu: Dict[FrozenSet[int], List[Tuple[float, float]]],
    sharing_rules: Dict[FrozenSet[int], Dict[int, float]],
) -> Dict[Tuple[FrozenSet[int], int], float]:
    """
    Calculate expected utilities for all players in all coalitions.
    """
    players = list(range(1, N + 1))
    coalitions = generate_coalitions(players)
    expected_utils = {}

    for coalition in coalitions:
        mu_S = mu.get(coalition, [])
        sharing_rule = sharing_rules.get(coalition, {})
        for player in coalition:
            eu = expected_utility(player, coalition, mu_S, sharing_rule, eta)
            expected_utils[(coalition, player)] = eu

    return expected_utils


def is_grand_coalition_stable(
    N: int,
    eta: float,
    mu: Dict[FrozenSet[int], List[Tuple[float, float]]],
    grand_sharing_rule: Dict[int, float],
    step_size: float = 0.1,
) -> bool:
    """
    Determine if the grand coalition is stable by checking all possible subset coalitions,
    including singletons, for possible deviations.

    :param N: Number of players.
    :param eta: Risk aversion parameter.
    :param mu: Characteristic function.
    :param grand_sharing_rule: Sharing rule for the grand coalition.
    :param step_size: Step size for iterating over sharing rules in subset coalitions.
    :return: True if stable, False otherwise.
    """
    players = list(range(1, N + 1))
    coalitions = generate_coalitions(players)
    grand_coalition = frozenset(players)

    # Define sharing rules: only grand coalition has a variable sharing rule
    # Singleton coalitions have fixed sharing rules: 100% to themselves
    sharing_rules = {coalition: {player: 1.0 for player in coalition} for coalition in coalitions if len(coalition) == 1}
    sharing_rules[grand_coalition] = grand_sharing_rule

    # Calculate expected utilities in the grand coalition
    expected_utils = calculate_all_expected_utilities(N, eta, mu, sharing_rules)
    grand_utils = {player: expected_utils.get((grand_coalition, player), 0) for player in grand_coalition}

    # Iterate over all possible subset coalitions, including singletons
    for subset_coalition in coalitions:
        if subset_coalition == grand_coalition:
            continue  # Skip the grand coalition itself

        subset_players = list(subset_coalition)

        if len(subset_coalition) == 1:
            # Singleton coalition: sharing rule is fixed (100% to themselves)
            player = subset_players[0]
            # Utility in subset coalition
            subset_utils = {}
            mu_S = mu.get(subset_coalition, [])
            sharing_rule = {player: 1.0}
            subset_utils[player] = expected_utility(player, subset_coalition, mu_S, sharing_rule, eta)

            # Compare with utility in grand coalition
            if subset_utils[player] > grand_utils[player]:
                # Player can achieve higher utility by deviating alone
                return False  # Grand coalition is not stable

        else:
            # Multi-player coalition: iterate over possible sharing rules
            num_steps = int(1 / step_size) + 1
            # Generate all possible sharing rules for the subset coalition
            # For simplicity, we assume that the sum of shares equals 1
            # Distribute the share to the first player, and the rest to others proportionally
            # To cover all possible sharing rules, especially for 2-player coalitions
            if len(subset_coalition) == 2:
                # For 2-player coalitions, vary the share of the first player
                for i in range(num_steps + 1):
                    alpha = i * step_size
                    if alpha > 1:
                        continue
                    # Define sharing rule
                    sharing_rule = {subset_players[0]: alpha, subset_players[1]: 1 - alpha}

                    # Update sharing rules
                    sharing_rules_subset = sharing_rules.copy()
                    sharing_rules_subset[subset_coalition] = sharing_rule

                    # Calculate expected utilities with this subset sharing rule
                    expected_utils_subset = calculate_all_expected_utilities(N, eta, mu, sharing_rules_subset)
                    subset_utils = {player: expected_utils_subset.get((subset_coalition, player), 0) for player in subset_coalition}

                    # Compare with utilities in grand coalition
                    all_better_or_equal = all(subset_utils[player] >= grand_utils[player] for player in subset_coalition)
                    at_least_one_better = any(subset_utils[player] > grand_utils[player] for player in subset_coalition)

                    if all_better_or_equal and at_least_one_better:
                        # This subset coalition can improve, grand coalition is not stable
                        return False

            else:
                # For coalitions with more than 2 players, similar logic can be applied
                # Here, we limit to 2-player coalitions for simplicity
                # You can extend this to larger coalitions if needed
                pass  # Implement if needed

    # If no subset coalition can improve, grand coalition is stable
    return True


def generate_simplex_grid(step_size):
    """
    Generate a grid over the simplex for three players.
    """
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


def ternary_to_cartesian(alpha1, alpha2, alpha3):
    """
    Convert ternary coordinates to Cartesian coordinates for plotting.
    """
    x = alpha2 + 0.5 * alpha3
    y = (np.sqrt(3) / 2) * alpha3
    return x, y


def plot_ternary_diagram(eta, stable_points, unstable_points):
    """
    Plot the stable and unstable points on a ternary diagram using matplotlib,
    and add interactivity to display (alpha1, alpha2, alpha3) on hover.
    """
    # Prepare data
    stable_points_cartesian = np.array([ternary_to_cartesian(*p) for p in stable_points])
    unstable_points_cartesian = np.array([ternary_to_cartesian(*p) for p in unstable_points])

    # Plotting
    plt.figure(figsize=(10, 9))

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
            s=30,
            alpha=0.6,
            edgecolors="w",
            linewidth=0.5,
        )

    # Plot unstable points
    if len(unstable_points_cartesian) > 0:
        scatter_unstable = plt.scatter(
            unstable_points_cartesian[:, 0], unstable_points_cartesian[:, 1], color="red", marker="x", label="Unstable", s=30, alpha=0.6
        )

    # Labels and title
    plt.title(f"Stable Sharing Rules in the Grand Coalition; eta={eta}", fontsize=16)
    plt.legend(loc="upper right")

    # Adjust plot limits and aspect
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, np.sqrt(3) / 2 + 0.05)
    plt.gca().set_aspect("equal", adjustable="box")

    # Remove ticks
    plt.xticks([])
    plt.yticks([])

    # Add labels to the corners
    offset = 0.05
    plt.text(-offset, -offset, "Alpha1", ha="center", va="center", fontsize=12)
    plt.text(1 + offset, -offset, "Alpha2", ha="center", va="center", fontsize=12)
    plt.text(0.5, np.sqrt(3) / 2 + offset, "Alpha3", ha="center", va="center", fontsize=12)

    # Remove the square box (axes frame)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

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
                text=f"{label}\nAlpha1: {alpha1:.2f}\nAlpha2: {alpha2:.2f}\nAlpha3: {alpha3:.2f}",
                position=(0, 20),
                anncoords="offset points",
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

    plt.show()


def main():
    N = 3
    eta = 1  # Risk aversion parameter (you can adjust this)
    grid_step_size = 0.02  # Adjust the step size for the grid (smaller step size for finer grid)
    step_size = 0.002  # For the stability checking
    players = list(range(1, N + 1))
    coalitions = generate_coalitions(players)

    # Define the characteristic function mu with stochastic payoffs for the grand coalition
    mu = {
        frozenset({1}): [(1.0, 0.5), (1.0, 0.5)],
        frozenset({2}): [(1.0, 0.5), (3.0, 0.5)],
        frozenset({3}): [(1.0, 0.5), (10.0, 0.5)],
        frozenset({1, 2}): [(5.0, 1.0)],
        frozenset({1, 3}): [(6.0, 1.0)],
        frozenset({2, 3}): [(5.0, 1.0)],
        frozenset({1, 2, 3}): [(10.0, 0.7), (15.0, 0.3)],
    }

    # Define sharing rules for singleton and pair coalitions
    # Note: We do NOT pre-define sharing rules for 2-player coalitions
    # because we need to check all possible sharing rules for them
    sharing_rules = {
        frozenset({1}): {1: 1.0},
        frozenset({2}): {2: 1.0},
        frozenset({3}): {3: 1.0},
        # 2-player coalitions will have their sharing rules checked dynamically
    }

    # Generate the simplex grid for the grand coalition sharing rules
    simplex_points = generate_simplex_grid(grid_step_size)

    # Prepare data for plotting
    stable_points = []
    unstable_points = []

    total_points = len(simplex_points)
    print(f"Total grand coalition sharing rules to evaluate: {total_points}")
    processed = 0

    for alpha1, alpha2, alpha3 in tqdm(simplex_points):
        # Define the sharing rule for the grand coalition
        grand_sharing_rule = {1: alpha1, 2: alpha2, 3: alpha3}

        # Check if the grand coalition is stable
        stable = is_grand_coalition_stable(N, eta, mu, grand_sharing_rule, step_size=step_size)

        # Collect the points
        if stable:
            stable_points.append((alpha1, alpha2, alpha3))
        else:
            unstable_points.append((alpha1, alpha2, alpha3))
    # Plot the results with interactivity
    plot_ternary_diagram(eta, stable_points, unstable_points)


if __name__ == "__main__":
    main()
