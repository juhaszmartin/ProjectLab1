import itertools
from typing import List, Tuple, Dict, FrozenSet
import numpy as np


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
        share = sharing_rule.get(player, 0.0) * payoff
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
            raise ValueError(f"Negative share encountered: {share}")
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

    for player in players:
        eu_grand = grand_utils[player]
        # Find maximum expected utility in any other coalition
        max_eu_other = float("-inf")
        for coalition in coalitions:
            if player in coalition and coalition != grand_coalition:
                eu = expected_utils.get((coalition, player), float("-inf"))
                if eu > max_eu_other:
                    max_eu_other = eu
        # Compare grand coalition utility with max in other coalitions
        if eu_grand < max_eu_other:
            stable = False
            report = "Grand coalition is NOT stable.\n"
            report += f"Player {player} can achieve higher utility ({max_eu_other:.4f}) in another coalition than in the grand coalition ({eu_grand:.4f}).\n"

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


def main():
    N = 2
    eta = 1  # Risk aversion parameter (adjust as needed)

    step_size = 0.05
    # Define ranges
    singleton_payoffs = np.arange(1, 2.05, step_size)  # From 0.1 to 1.0
    grand_payoffs = np.arange(1, 3.05, step_size)  # From 0.1 to 2.0
    alpha_values = np.arange(0, 1.05, step_size)  # From 0.1 to 0.9

    # Initialize a dictionary to store stable alphas for each payoff combination
    stable_alpha_dict = {}

    # Loop over all combinations
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
                    payoff_key = (v1, v2, V)
                    if payoff_key not in stable_alpha_dict:
                        stable_alpha_dict[payoff_key] = []
                    if stable:
                        stable_alpha_dict[payoff_key].append(alpha1)

    # After collecting all stable alphas, print the results
    print("\n=== Range of alpha1 Values for Grand Coalition Stability ===\n")
    for payoff_key, alphas in sorted(stable_alpha_dict.items()):
        v1, v2, V = payoff_key
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
            print(f"Payoffs: v1={v1:.2f}, v2={v2:.2f}, V={V:.2f} => Stable alpha1 Range(s): {range_str}")
        else:
            print(f"Payoffs: v1={v1:.2f}, v2={v2:.2f}, V={V:.2f} => No stable alpha1 values.")


if __name__ == "__main__":
    main()
