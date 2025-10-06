# Starter code 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt


# Problem setup
# Update to point to the dataset on your machine
data: pd.DataFrame = pd.read_csv("/Users/winwin/Documents/School/CPP/CS5510/HW1/fake_healthcare_dataset_sample100.csv")

# names of public identifier columns
pub = ["age", "sex", "blood", "admission"]

# variable to reconstruct
target = "result"


def execute_subsetsums_exact(predicates):
    """Count the number of patients that satisfy each predicate.
    Resembles a public query interface on a sequestered dataset.
    Computed as in equation (1).

    :param predicates: a list of predicates on the public variables
    :returns a 1-d np.ndarray of exact answers the subset sum queries"""
    return data[target].values @ np.stack([pred(data) for pred in predicates], axis=1)

# TODO: Implement these defense mechanisms
def execute_subsetsums_round(R,predicates):
    """
    Round each subset-sum answer to the nearest multiple of R.
    """
    exact = execute_subsetsums_exact(predicates)
    noisy = R * np.round(exact / R)
    return noisy

def execute_subsetsums_noise(sigma,predicates):
    """
    Add independent Gaussian noise with mean 0 and std-dev sigma [ N(0, sigma^2 ] to each answer.s
    """
    exact = execute_subsetsums_exact(predicates)
    noisy = exact + np.random.normal(scale=sigma, size=len(exact))
    return noisy

def execute_subsetsums_sample(t,predicates):
    """
    Randomly subsample t rows (without replacement), compute answers on that subset
    then scale by n/t.
    """
    n = len(data)
    sampled_indices = np.random.choice(n, size=t, replace=False)
    sampled_data = data.iloc[sampled_indices]
    return sampled_data[target].values @ np.stack([pred(sampled_data) for pred in predicates], axis=1) * (n / t)

def make_random_predicate():
    """Returns a (pseudo)random predicate function by hashing public identifiers."""
    prime = 2003
    desc = np.random.randint(prime, size=len(pub))
    # this predicate maps data into a 1-d ndarray of booleans
    #   (where `@` is the dot product and `%` modulus)
    return lambda data: ((data[pub].values @ desc) % prime % 2).astype(bool)

# TODO: Write the reconstruction function!
def reconstruction_attack(data_pub, predicates, answers):
    """Reconstructs a target column based on the `answers` to queries about `data`.

    :param data_pub: data of length n consisting of public identifiers
    :param predicates: a list of k predicate functions
    :param answers: a list of k answers to a query on data filtered by the k predicates
    :return 1-dimensional boolean ndarray"""
    n = len(data_pub)

    # Build Q (n x k): each column is the boolean mask for one predicate
    masks = []
    for j, pred in enumerate(predicates):
        mask = pred(data_pub)                        # could be Series or ndarray
        mask = np.asarray(mask, dtype=bool).ravel()  # make 1D boolean array
        if mask.shape[0] != n:
            raise ValueError(f"Predicate {j} produced length {mask.shape[0]} != n={n}")
        masks.append(mask)

    Q = np.column_stack(masks).astype(float)        # shape: (n, k)
    b = np.asarray(answers, dtype=float).ravel()    # shape: (k,)

    # sanity check
    if Q.shape[1] != b.shape[0]:
        raise ValueError(f"Got {Q.shape[1]} predicates but {b.shape[0]} answers")
    
    # Solve the least squares problem to get a real-valued solution x_hat
    x_hat, *_ = np.linalg.lstsq(Q.T, b, rcond=None)

    # Project to {0,1}
    x_hat = np.clip(x_hat, 0.0, 1.0)
    return (x_hat >= 0.5)


# ===================== Experiment =====================
def rmse(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return np.sqrt(np.mean((a - b) ** 2))

def run_one_trial(defense_fn, param_value, k, data_pub, truth_bits):
    """
    Runs one trial:
      1) sample k random predicates
      2) get exact answers and defended answers
      3) compute RMSE and reconstruction success
    defense_fn: callable(predicates) -> defended answers (depends on param_value closed over by lambda)
    """
    # fresh randomness per trial via default RNG
    predicates = [make_random_predicate() for _ in range(k)]

    exact_answers = execute_subsetsums_exact(predicates)
    defended_answers = defense_fn(predicates)

    # accuracy of answers
    answers_rmse = rmse(defended_answers, exact_answers)

    # reconstruction success
    recon_bits = reconstruction_attack(data_pub, predicates, defended_answers)
    success = (recon_bits == truth_bits).mean()

    return answers_rmse, success

def sweep_param(defense_name, build_defense_callable, param_values, reps, data_pub, truth_bits):
    """
    For each parameter value, run `reps` trials and average RMSE and success.
    build_defense_callable: function(param) -> callable(predicates)->answers
    """
    n_params = len(param_values)
    avg_rmse = np.zeros(n_params, dtype=float)
    avg_success = np.zeros(n_params, dtype=float)

    k = 2 * len(data_pub)  # follow assignment: use 2n random predicates

    for idx, p in enumerate(param_values):
        rmse_list = []
        succ_list = []
        for _ in range(reps):
            defense_fn = build_defense_callable(p)
            a_rmse, s = run_one_trial(defense_fn, p, k, data_pub, truth_bits)
            rmse_list.append(a_rmse)
            succ_list.append(s)
        avg_rmse[idx] = np.mean(rmse_list)
        avg_success[idx] = np.mean(succ_list)
        print(f"[{defense_name}] param={p:>3} | RMSE={avg_rmse[idx]:.3f} | success={avg_success[idx]:.3f}")

    return avg_rmse, avg_success

def plot_param_curves(param_values, avg_rmse, avg_success, defense_label, majority_frac):
    # 1) Parameter vs RMSE
    plt.figure()
    plt.plot(param_values, avg_rmse, label=f"{defense_label} RMSE")
    plt.xlabel("Parameter value")
    plt.ylabel("Answer RMSE (vs exact)")
    plt.title(f"Answer Accuracy vs Parameter — {defense_label}")
    plt.legend()
    plt.tight_layout()

    # 2) Parameter vs Reconstruction Success
    plt.figure()
    plt.plot(param_values, avg_success, label=f"{defense_label} Success")
    # majority baseline line
    plt.axhline(majority_frac)
    plt.xlabel("Parameter value")
    plt.ylabel("Reconstruction success (fraction correct)")
    plt.title(f"Reconstruction Success vs Parameter — {defense_label}")
    plt.legend()
    plt.tight_layout()

def plot_tradeoff(avg_rmse, avg_success, defense_label):
    # 3) Trade-off: RMSE (x) vs Success (y)
    plt.figure()
    plt.plot(avg_rmse, avg_success, marker="o", linestyle="-", label=defense_label)
    plt.xlabel("Answer RMSE (lower is more accurate)")
    plt.ylabel("Reconstruction success (higher is better for attacker)")
    plt.title(f"Accuracy–Success Trade-off — {defense_label}")
    plt.legend()
    plt.tight_layout()

def find_transition(param_values, avg_success, majority_frac, near_perfect=0.98, tol=0.01):
    """
    Identify a rough 'transition' regime:
      - smallest param where success <= majority_frac + tol (attack basically fails)
      - largest param where success >= near_perfect (attack is near-perfect)
    """
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
    num_predicates = 2 * num_patients  # for k = 2n random predicates
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

    # (c) Sweep parameters for each defense mechanism
    print("\n====== Parameter sweeps for defenses ======")
    # Public view and truth bits
    data_pub = data[pub].copy()
    truth_bits = data[target].values.astype(bool)
    n = len(data_pub)

    # Majority baseline (no-info classifier)
    majority_frac = max(truth_bits.mean(), 1.0 - truth_bits.mean())
    print(f"Majority baseline fraction: {majority_frac:.3f}")

    # Parameter grids and repetitions
    param_values = list(range(1, n + 1))  # 1..n
    reps = 10

    # Build callables that close over parameter p
    round_builder  = lambda p: (lambda preds: execute_subsetsums_round(p, preds))
    noise_builder  = lambda p: (lambda preds: execute_subsetsums_noise(p, preds))
    sample_builder = lambda p: (lambda preds: execute_subsetsums_sample(p, preds))

    # ----- Run sweeps -----
    print("\n=== Sweeping R (rounding) ===")
    rmse_round, succ_round = sweep_param("round", round_builder, param_values, reps, data_pub, truth_bits)

    print("\n=== Sweeping sigma (Gaussian noise) ===")
    rmse_noise, succ_noise = sweep_param("noise", noise_builder, param_values, reps, data_pub, truth_bits)

    print("\n=== Sweeping t (subsample size) ===")
    rmse_sample, succ_sample = sweep_param("sample", sample_builder, param_values, reps, data_pub, truth_bits)

    # ----- Plot per-defense curves -----
    plot_param_curves(param_values, rmse_round,  succ_round,  "Rounding (R)",          majority_frac)
    plot_param_curves(param_values, rmse_noise,  succ_noise,  "Gaussian Noise (σ)",     majority_frac)
    plot_param_curves(param_values, rmse_sample, succ_sample, "Subsample (t of n)",     majority_frac)

    # ----- Trade-off plots (RMSE vs Success) -----
    plot_tradeoff(rmse_round,  succ_round,  "Rounding (R)")
    plot_tradeoff(rmse_noise,  succ_noise,  "Gaussian Noise (σ)")
    plot_tradeoff(rmse_sample, succ_sample, "Subsample (t of n)")

    # ----- Print transition points -----
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

    # Show all figures at the end
    plt.show()


    
