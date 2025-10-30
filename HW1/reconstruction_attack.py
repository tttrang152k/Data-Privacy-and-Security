# Starter code 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# Problem setup
data: pd.DataFrame = pd.read_csv("/Users/winwin/Documents/School/CPP/CS5510/HW1/fake_healthcare_dataset_sample100.csv")
pub = ["age", "sex", "blood", "admission"]
target = "result"

def execute_subsetsums_exact(predicates):
    return data[target].values @ np.stack([pred(data) for pred in predicates], axis=1)

# Implement defense mechanisms
def execute_subsetsums_round(R, predicates):
    """
    Round each subset-sum answer to the nearest multiple of R.
    """
    exact = execute_subsetsums_exact(predicates)
    return R * np.round(exact / R)

def execute_subsetsums_noise(sigma, predicates):
    """
    Add independent Gaussian noise with mean 0 and std-dev sigma [ N(0, sigma^2 ] to each answer
    """
    exact = execute_subsetsums_exact(predicates)
    return exact + np.random.normal(scale=sigma, size=len(exact))

def execute_subsetsums_sample(t, predicates):
    """
    Randomly subsample t rows (without replacement), compute answers on that subset
    then scale by n/t.
    """
    n = len(data)
    idx = np.random.choice(n, size=t, replace=False)
    sub = data.iloc[idx]
    return sub[target].values @ np.stack([pred(sub) for pred in predicates], axis=1) * (n / t)

def make_random_predicate():
    """Returns a (pseudo)random predicate function by hashing public identifiers."""
    prime = 2003
    desc = np.random.randint(prime, size=len(pub))
    return lambda data: ((data[pub].values @ desc) % prime % 2).astype(bool)

# TODO: Write the reconstruction function!
def reconstruction_attack(data_pub, predicates, answers):
    """Reconstructs a target column based on the `answers` to queries about `data`.

    :param data_pub: data of length n consisting of public identifiers
    :param predicates: a list of k predicate functions
    :param answers: a list of k answers to a query on data filtered by the k predicates
    :return 1-dimensional boolean ndarray"""
    n = len(data_pub)
    masks = []
    for j, pred in enumerate(predicates):
        mask = np.asarray(pred(data_pub), dtype=bool).ravel()
        if mask.shape[0] != n:
            raise ValueError(f"Predicate {j} produced length {mask.shape[0]} != n={n}")
        masks.append(mask)
    Q = np.column_stack(masks).astype(float)     # n x k
    b = np.asarray(answers, dtype=float).ravel() # k,
    if Q.shape[1] != b.shape[0]:
        raise ValueError(f"Got {Q.shape[1]} predicates but {b.shape[0]} answers")
    x_hat, *_ = np.linalg.lstsq(Q.T, b, rcond=None)
    x_hat = np.clip(x_hat, 0.0, 1.0)
    return (x_hat >= 0.5)

# Experiment 
def rmse(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return np.sqrt(np.mean((a - b) ** 2))

def run_one_trial(defense_fn, param_value, k, data_pub, truth_bits):
    predicates = [make_random_predicate() for _ in range(k)]
    exact_answers = execute_subsetsums_exact(predicates)
    defended_answers = defense_fn(predicates)
    answers_rmse = rmse(defended_answers, exact_answers)
    recon_bits = reconstruction_attack(data_pub, predicates, defended_answers)
    success = (recon_bits == truth_bits).mean()
    return answers_rmse, success

def evenly_spaced_indices(n, m=10):
    """Pick m (≤ n) evenly spaced indices in [0, n-1]."""
    m = min(m, n)
    return sorted(set(np.linspace(0, n-1, m, dtype=int).tolist()))

def sweep_param(defense_name, build_defense_callable, param_values, reps, data_pub, truth_bits, print_count=10):
    """
    Run reps trials for each parameter; return avg RMSE & success.
    Only PRINT results for a subset of 'print_count' evenly spaced parameter values.
    """
    n_params = len(param_values)
    avg_rmse = np.zeros(n_params, dtype=float)
    avg_success = np.zeros(n_params, dtype=float)

    k = 2 * len(data_pub)  # follow assignment: use 2n random predicates

    # choose which parameter indices to print
    print_idxs = set(evenly_spaced_indices(n_params, m=print_count))

    for idx, p in enumerate(param_values):
        rmse_list, succ_list = [], []
        for _ in range(reps):
            defense_fn = build_defense_callable(p)
            a_rmse, s = run_one_trial(defense_fn, p, k, data_pub, truth_bits)
            rmse_list.append(a_rmse)
            succ_list.append(s)
        avg_rmse[idx] = np.mean(rmse_list)
        avg_success[idx] = np.mean(succ_list)

        if idx in print_idxs:
            print(f"[{defense_name}] param={p:>3} | RMSE={avg_rmse[idx]:.3f} | success={avg_success[idx]:.3f}")

    return avg_rmse, avg_success

# RMSE + Success plots
def plot_param_curves_combined(param_values, avg_rmse, avg_success, defense_label, majority_frac):
    fig, ax1 = plt.subplots()

    # RMSE (left y-axis) 
    rmse_line, = ax1.plot(
        param_values, avg_rmse,
        color="red", linewidth=2, label="RMSE"
    )
    ax1.set_xlabel("Parameter value")
    ax1.set_ylabel("Answer RMSE (vs exact)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.grid(True, alpha=0.25)

    # Success (right y-axis)
    ax2 = ax1.twinx()
    succ_line, = ax2.plot(
        param_values, avg_success,
        color="blue", linewidth=2, label="Success"
    )
    ax2.set_ylabel("Reconstruction success (fraction correct)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Majority baseline
    base = ax2.axhline(
        majority_frac, color="green", linewidth=2, label="Majority baseline"
    )

    # Title + unified legend
    lines = [rmse_line, succ_line, base]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    plt.title(f"{defense_label}: RMSE & Success vs Parameter")
    plt.tight_layout()

def plot_tradeoff(avg_rmse, avg_success, defense_label):
    plt.figure()
    plt.plot(avg_rmse, avg_success, marker="o", linestyle="-", label=defense_label)
    plt.xlabel("Answer RMSE (lower is more accurate)")
    plt.ylabel("Reconstruction success (higher is better for attacker)")
    plt.title(f"Accuracy–Success Trade-off — {defense_label}")
    plt.legend()
    plt.tight_layout()

def find_transition(param_values, avg_success, majority_frac, near_perfect=0.98, tol=0.01):
    fail_threshold = majority_frac + tol
    idx_fail = next((i for i, s in enumerate(avg_success) if s <= fail_threshold), None)
    idx_perfect = max([i for i, s in enumerate(avg_success) if s >= near_perfect], default=None)
    return (
        None if idx_perfect is None else param_values[idx_perfect],
        None if idx_fail is None else param_values[idx_fail],
    )

if __name__ == "__main__":
    # (a) Complete the reconstruction attack
    data_pub = data[pub].copy()
    num_patients = len(data_pub)
    num_predicates = 2 * num_patients   # k = 2n predicates

    predicates = [make_random_predicate() for _ in range(num_predicates)]
    # exact answers from the public query interface
    answers = execute_subsetsums_exact(predicates)
    # reconstructed target column
    reconstructed = reconstruction_attack(data_pub, predicates, answers)

    # accuracy of the reconstruction
    print("------ Testing k = 2n values ------")
    truth = data[target].values.astype(bool)
    acc = (reconstructed == truth).mean()
    num_errors = (reconstructed != truth).sum()
    print(f"Accuracy: {acc:.3f}  |  Errors: {num_errors} out of {num_patients}")

    # TESTING purposes: experiment with different numbers of predicates k < n
    print("------ Testing different k values ------")
    for k in [10, 20, 50, 75, 100, 200, 300]:
        predicates = [make_random_predicate() for _ in range(k)]
        answers = execute_subsetsums_exact(predicates)
        reconstructed = reconstruction_attack(data_pub, predicates, answers)
        acc = (reconstructed == truth).mean()
        num_errors = (reconstructed != truth).sum()
        print(f"k={k:3d}  |  Accuracy: {acc:.3f}  |  Errors: {num_errors} out of {num_patients}")

    # (b) Experiment with the defense mechanisms
    predicates = [make_random_predicate() for _ in range(2 * len(data))]

    # exact (baseline)
    answers_exact = execute_subsetsums_exact(predicates)
    reconstructed_exact = reconstruction_attack(data_pub, predicates, answers_exact)
    acc_exact = (reconstructed_exact == truth).mean()
    print(f"Exact answers accuracy: {acc_exact:.3f}")

    # rounding: e.g., nearest multiple of 5
    answers_round = execute_subsetsums_round(R=5, predicates=predicates)
    reconstructed_round = reconstruction_attack(data_pub, predicates, answers_round)
    acc_round = (reconstructed_round == truth).mean()
    print(f"Rounding answers accuracy: {acc_round:.3f}")

    # Gaussian noise: e.g., sigma = 2.0
    answers_noise = execute_subsetsums_noise(sigma=2.0, predicates=predicates)
    reconstructed_noise = reconstruction_attack(data_pub, predicates, answers_noise)
    acc_noise = (reconstructed_noise == truth).mean()
    print(f"Gaussian noise answers accuracy: {acc_noise:.3f}")

    # subsample: e.g., t = 40 rows out of n
    answers_sample = execute_subsetsums_sample(t=40, predicates=predicates)
    reconstructed_sample = reconstruction_attack(data_pub, predicates, answers_sample)
    acc_sample = (reconstructed_sample == truth).mean()
    print(f"Subsample answers accuracy: {acc_sample:.3f}")

    # (c) Experiment parameters for each defense mechanism
    print("\n====== Parameter experiment for defenses ======")
    truth_bits = data[target].values.astype(bool)
    n = len(data_pub)
    majority_frac = max(truth_bits.mean(), 1.0 - truth_bits.mean())
    print(f"Majority baseline fraction: {majority_frac:.3f}")

    param_values = list(range(1, n + 1))  # 1..n
    reps = 10

    round_builder  = lambda p: (lambda preds: execute_subsetsums_round(p, preds))
    noise_builder  = lambda p: (lambda preds: execute_subsetsums_noise(p, preds))
    sample_builder = lambda p: (lambda preds: execute_subsetsums_sample(p, preds))

    print("\n=== Experimenting R (rounding) ===")
    rmse_round,  succ_round  = sweep_param("round",  round_builder,  param_values, reps, data_pub, truth_bits, print_count=10)

    print("\n=== Experimenting σ (Gaussian noise) ===")
    rmse_noise,  succ_noise  = sweep_param("noise",  noise_builder,  param_values, reps, data_pub, truth_bits, print_count=10)

    print("\n=== Experimenting t (subsample size) ===")
    rmse_sample, succ_sample = sweep_param("sample", sample_builder, param_values, reps, data_pub, truth_bits, print_count=10)

    # RMSE & Success combined plots 
    plot_param_curves_combined(param_values, rmse_round,  succ_round,  "Rounding (R)",          majority_frac)
    plot_param_curves_combined(param_values, rmse_noise,  succ_noise,  "Gaussian Noise (σ)",     majority_frac)
    plot_param_curves_combined(param_values, rmse_sample, succ_sample, "Subsample (t of n)",     majority_frac)

    # Trade-off plots
    plot_tradeoff(rmse_round,  succ_round,  "Rounding (R)")
    plot_tradeoff(rmse_noise,  succ_noise,  "Gaussian Noise (σ)")
    plot_tradeoff(rmse_sample, succ_sample, "Subsample (t of n)")

    # Regime 
    rp, rf = find_transition(param_values, succ_round,  majority_frac)
    np_, nf = find_transition(param_values, succ_noise,  majority_frac)
    sp, sf = find_transition(param_values, succ_sample, majority_frac)

    print("\n=== Transition summary (approximate) ===")
    if rp is not None: print(f"Rounding: last near-perfect ≥98% at R≈{rp}")
    if rf is not None: print(f"Rounding: first fail (≤ majority) at R≈{rf}")
    if np_ is not None: print(f"Noise:    last near-perfect ≥98% at σ≈{np_}")
    if nf is not None: print(f"Noise:    first fail (≤ majority) at σ≈{nf}")
    if sp is not None: print(f"Sample:   last near-perfect ≥98% at t≈{sp}")
    if sf is not None: print(f"Sample:   first fail (≤ majority) at t≈{sf}")

    plt.show()
