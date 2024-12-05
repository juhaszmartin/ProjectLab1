import itertools
from typing import List, Tuple, Dict, FrozenSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache


def generate_coalitions(players: List[int]) -> List[FrozenSet[int]]:
    """
    Generate all possible coalitions (subsets) excluding the empty set.

    :param players: List of player identifiers.
    :return: List of coalitions as frozensets.
    """
    coalitions = []
    for r in range(1, len(players) + 1):
        coalitions.extend([frozenset(c) for c in itertools.combinations(players, r)])
    return coalitions


# @lru_cache
def expected_utility(
    player: int,
    coalition: FrozenSet[int],
    mu_S: List[Tuple[float, float]],
    sharing_rule: Dict[int, float],
    eta: float,
) -> float:
    exp_util = 0.0
    for payoff, prob in mu_S:
        # Allocate share based on sharing rule
        share = (
            sharing_rule.get(
                player,
            )
            * payoff
        )
        # print(f"payoff: {payoff}")
        # print(f"prob: {prob}")
        # print(f"player: {player}")
        # print(f"sharing_rule: {sharing_rule}")
        # print(f"share: {share}")
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
            raise ValueError(
                f"Negative share encountered: {share}, {payoff}, {sharing_rule.get(
                player,
            )}"
            )
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

    :param N: Number of players.
    :param eta: Risk aversion parameter.
    :param mu: Characteristic function mapping coalitions to payoff distributions.
    :param sharing_rules: Sharing rules mapping coalitions to player share proportions.
    :return: Dictionary mapping (coalition, player) to expected utility.
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
    sharing_rules: Dict[FrozenSet[int], Dict[int, float]],
) -> Tuple[bool, str]:
    """
    Determine if the grand coalition is stable.

    :param N: Number of players.
    :param eta: Risk aversion parameter.
    :param mu: Characteristic function.
    :param sharing_rules: Sharing rules.
    :return: Tuple indicating stability and a report.
    """
    players = list(range(1, N + 1))
    coalitions = generate_coalitions(players)
    grand_coalition = frozenset(players)

    # Calculate expected utilities
    expected_utils = calculate_all_expected_utilities(N, eta, mu, sharing_rules)

    # Extract expected utilities in the grand coalition
    grand_utils = {player: expected_utils.get((grand_coalition, player), 0) for player in grand_coalition}

    stable = True
    report = "Grand coalition is stable.\n"

    epsilon = 1e-8  # Tolerance for floating-point comparisons

    for player in players:
        eu_grand = grand_utils[player]
        # Find maximum expected utility in any other coalition
        max_eu_other = float("-inf")
        for coalition in coalitions:
            if player in coalition and coalition != grand_coalition:
                eu = expected_utils.get((coalition, player), float("-inf"))
                if eu > max_eu_other:
                    max_eu_other = eu
        # Compare grand coalition utility with max in other coalitions using epsilon
        if eu_grand + epsilon < max_eu_other:
            stable = False
            report = f"Grand coalition is NOT stable.\n"
            report += f"Player {player} can achieve higher utility ({max_eu_other:.4f}) in another coalition than in the grand coalition ({eu_grand:.4f}).\n"
            # Break early since we've found instability
            break

    return stable, report


def print_expected_utilities(expected_utils: Dict[Tuple[FrozenSet[int], int], float]):
    """
    Print the expected utilities in a readable format.

    :param expected_utils: Dictionary mapping (coalition, player) to expected utility.
    """
    print("Expected Utilities:")
    for key in sorted(expected_utils.keys(), key=lambda x: (len(x[0]), sorted(x[0]), x[1])):
        coalition, player = key
        util = expected_utils[key]
        print(f"  Coalition {set(coalition)}, Player {player}: Expected Utility = {util:.4f}")


def visualize_stable_alpha_ranges(df, parameter_combinations):
    """
    Generate a grid of plots showing the stable alpha ranges for eta=1 and eta=10.
    Keep titles on all plots and remove x-axis numbers except on the bottom plots.
    Adjust subplot parameters as specified.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    eta_values = [1, 10]  # The two eta values to compare

    num_params = len(parameter_combinations)
    # Create a figure with n rows and 2 columns
    fig, axes = plt.subplots(nrows=num_params, ncols=2, figsize=(12, num_params * 2.5))

    # Adjust the spacing and margins between subplots using your specified parameters
    plt.subplots_adjust(top=0.936, bottom=0.141, left=0.024, right=0.976, hspace=0.55, wspace=0.071)

    for idx, (v1, v2, V) in enumerate(parameter_combinations):
        for col_idx, eta in enumerate(eta_values):
            ax = axes[idx, col_idx]
            df_subset = df[(df["v1"] == v1) & (df["v2"] == v2) & (df["V"] == V) & (df["eta"] == eta) & (df["stable"] == True)]
            if df_subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No stable alpha1 values",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
            else:
                # Plot stable alpha1 values
                stable_alpha1 = df_subset["alpha1"]
                ax.plot(stable_alpha1, [0.5] * len(stable_alpha1), "o", color="blue")
            # Set titles with parameters
            ax.set_title(f"eta={eta}, v1={v1}, v2={v2}, V={V}", fontsize=10, pad=10)
            # Remove x-axis tick labels for non-bottom plots
            if idx != num_params - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("alpha1", fontsize=9)
            # Remove y-axis ticks
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    # Add a super title for the entire figure
    fig.suptitle("Stable alpha1 Values for Different Parameters", fontsize=16, y=0.99)
    # No need for plt.tight_layout() since we're using plt.subplots_adjust()
    plt.show()


def main_visual():
    N = 2
    eta_values = [1, 10]  # Values of eta to compare
    step_size = 0.01
    # singleton_payoffs = [1.0, 1.5, 2.0]  # Adjusted for selected combinations
    # grand_payoffs = [2.0, 2.5, 3.0]
    alpha_values = np.arange(0, 1.0 + step_size / 2, step_size)
    # Define 10 parameter combinations for v1, v2, V
    parameter_combinations = [
        (1.0, 1.0, 2.0),
        (1.0, 1.5, 2.5),
        (1.0, 2.0, 3.0),
        (1.5, 1.0, 2.5),
        (1.5, 1.5, 3.0),
        (2.0, 1.0, 3.0),
        (2.0, 2.0, 2.5),
        (1.0, 1.0, 5.0),
    ]

    data_records = []

    for eta in eta_values:
        print(f"Processing eta={eta}...")
        for v1, v2, V in parameter_combinations:
            for alpha1 in alpha_values:
                alpha2 = 1 - alpha1
                mu = {
                    frozenset({1}): [(1.0, 0.5), (v1, 0.5)],
                    frozenset({2}): [(1.0, 0.5), (v2, 0.5)],
                    frozenset({1, 2}): [(V, 1.0)],
                }
                sharing_rules = {
                    frozenset({1}): {1: 1.0},
                    frozenset({2}): {2: 1.0},
                    frozenset({1, 2}): {1: alpha1, 2: alpha2},
                }
                stable, _ = is_grand_coalition_stable(N, eta, mu, sharing_rules)
                data_records.append({"eta": eta, "v1": v1, "v2": v2, "V": V, "alpha1": alpha1, "stable": stable})

    df = pd.DataFrame(data_records)
    visualize_stable_alpha_ranges(df, parameter_combinations)


def main():
    N = 2
    # eta = 1  # Risk aversion parameter (adjust as needed)
    eta_values = np.arange(0.0, 11, step=1)  # From 0 to 10
    step_size = 0.05
    # Define ranges
    singleton_payoffs = np.arange(0.5, 5.15, step_size)  # From 0.1 to 2.0
    grand_payoffs = np.arange(4, 4.05, step_size)  # From 0.1 to 3.0
    alpha_values = np.arange(0, 1.05, step_size)  # From 0 to 1

    # Initialize a dictionary to store stable alphas for each payoff combination
    stable_alpha_dict = {}

    # Loop over all combinations
    for eta in eta_values:
        for v1 in singleton_payoffs:
            for v2 in singleton_payoffs:
                for V in grand_payoffs:
                    for alpha1 in alpha_values:
                        alpha2 = 1 - alpha1
                        # Define characteristic function mu
                        mu = {
                            frozenset({1}): [(1.0, 0.5), (v1, 0.5)],
                            frozenset({2}): [(1.0, 0.5), (v2, 0.5)],
                            frozenset({1, 2}): [(V, 1.0)],
                        }
                        # Define sharing rules
                        sharing_rules = {
                            frozenset({1}): {1: 1.0},
                            frozenset({2}): {2: 1.0},
                            frozenset({1, 2}): {1: alpha1, 2: alpha2},
                        }
                        # Calculate expected utilities
                        # expected_utils = calculate_all_expected_utilities(N, eta, mu, sharing_rules, shift_constant)
                        # Check stability
                        stable, report = is_grand_coalition_stable(N, eta, mu, sharing_rules)
                        # If stable, record alpha1 for this payoff combination
                        payoff_key = (eta, v1, v2, V)
                        if payoff_key not in stable_alpha_dict:
                            stable_alpha_dict[payoff_key] = []
                        if stable:
                            stable_alpha_dict[payoff_key].append(alpha1)

    # After collecting all stable alphas, print the results
    print("\n=== Range of alpha1 Values for Grand Coalition Stability ===\n")
    for payoff_key, alphas in sorted(stable_alpha_dict.items()):
        eta, v1, v2, V = payoff_key
        if alphas:
            alphas_sorted = sorted(alphas)
            # Determine continuous ranges
            ranges = []
            start = alphas_sorted[0]
            end = alphas_sorted[0]
            for alpha in alphas_sorted[1:]:
                if np.isclose(alpha, end + step_size, atol=1e-8):
                    end = alpha
                else:
                    ranges.append((start, end))
                    start = alpha
                    end = alpha

            ranges.append((start, end))
            # Format ranges for display
            range_str = ", ".join([f"{start:.2f}-{end:.2f}" if start != end else f"{start:.2f}" for start, end in ranges])
            print(f"Eta={eta} Payoffs: v1={v1:.2f}, v2={v2:.2f}, V={V:.2f} => Stable alpha1 Range(s): {range_str}")
        else:
            print(f"Eta={eta} Payoffs: v1={v1:.2f}, v2={v2:.2f}, V={V:.2f} => No stable alpha1 values.")


def test():
    step_size = 0.05
    alpha_values = np.array([0.2, 0.5, 0.8])  # From 0 to 1
    for alpha1 in alpha_values:
        alpha2 = 1 - alpha1
        mu = {
            frozenset({1}): [(1.0, 0.5), (1, 0.5)],
            frozenset({2}): [(1.0, 0.5), (1, 0.5)],
            frozenset({1, 2}): [(5, 1.0)],
        }
        # Define sharing rules
        sharing_rules = {
            frozenset({1}): {1: 1.0},
            frozenset({2}): {2: 1.0},
            frozenset({1, 2}): {1: alpha1, 2: alpha2},
        }
        # asd = expected_utility(
        #    player=1, coalition=frozenset({1, 2}), mu_S=mu.get(frozenset({1, 2}), []), sharing_rule={1: 0.45, 2: 0.55}, eta=1
        # )
        # print(asd)
        print(is_grand_coalition_stable(2, 1, mu, sharing_rules))


if __name__ == "__main__":
    main_visual()
