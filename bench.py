"""
Performance benchmark for all hypothesis tests.

Measures per-call latency at three input sizes (small / medium / large)
and reports mean, min, and max over REPS repetitions.

Run:
    python bench.py
"""

import time
import tracemalloc
import numpy as np

from app.hypothesis_engine import run_test, run_hypothesis

# ── Config ────────────────────────────────────────────────────────────────────
REPS        = 200   # repetitions per test/size combination
REPS_SLOW   = 20    # for tests known to be expensive at large N
SIZES       = {"small": 50, "medium": 500, "large": 5_000}

# Tests that use OLS or ruptures internally — cap reps at large N
SLOW_TESTS  = {"interrupted_time_series", "detect_change_point", "tukey_hsd",
               "linear_regression_test"}

# ── Color helpers (mirrors known_answer_suite.py) ─────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ── Data builders ─────────────────────────────────────────────────────────────

def series(n):
    np.random.seed(0)
    return list(np.linspace(0, 1, n) + np.random.normal(0, 0.05, n))

def two_groups(n):
    np.random.seed(0)
    return {"groups": [
        list(np.random.normal(100, 10, n)),
        list(np.random.normal(110, 10, n)),
    ]}

def three_groups(n):
    np.random.seed(0)
    return {"groups": [
        list(np.random.normal(100, 10, n)),
        list(np.random.normal(110, 10, n)),
        list(np.random.normal(120, 10, n)),
    ]}

def did_data(n):
    np.random.seed(0)
    return {
        "treatment_before": list(np.random.normal(50_000, 3_000, n)),
        "treatment_after":  list(np.random.normal(62_000, 3_000, n)),
        "control_before":   list(np.random.normal(48_000, 3_000, n)),
        "control_after":    list(np.random.normal(49_000, 3_000, n)),
    }

def its_data(n):
    half = n // 2
    np.random.seed(0)
    return {
        "series": list(np.random.normal(50_000, 2_000, half))
                + list(np.random.normal(62_000, 2_000, half)),
        "intervention_index": half,
    }

def cp_data(n):
    half = n // 2
    np.random.seed(0)
    return {"series": list(np.concatenate([
        np.random.normal(0, 1, half),
        np.random.normal(5, 1, half),
    ]))}

def ks_data(n):
    np.random.seed(0)
    return {
        "sample_a": list(np.random.normal(0, 1, n)),
        "sample_b": list(np.random.normal(1, 1, n)),
    }

def chi_data(n):
    np.random.seed(0)
    counts = list(np.random.randint(10, 100, n))
    return {"observed": counts}

FIXED_Z = {"current_value": 0.042, "historical_mean": 0.08, "historical_std": 0.01}


# ── Benchmark harness ─────────────────────────────────────────────────────────

def bench(test_name, data_fn, sizes=None, fixed_data=None):
    """
    Run `test_name` REPS times at each size in `sizes`, or once with
    `fixed_data` if the test doesn't scale with N.
    """
    targets = {"fixed": None} if fixed_data is not None else sizes

    rows = []
    for label, n in targets.items():
        data = fixed_data if fixed_data is not None else data_fn(n)

        reps = REPS_SLOW if (test_name in SLOW_TESTS and label == "large") else REPS
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            run_test(test_name, data)
            times.append((time.perf_counter() - t0) * 1_000)

        mean_ms = sum(times) / len(times)
        min_ms  = min(times)
        max_ms  = max(times)
        rows.append((label, n, mean_ms, min_ms, max_ms))

    return rows


def bench_pipeline(sizes):
    """Benchmark the full run_hypothesis pipeline (parse → select → run)."""
    rows = []
    for label, n in sizes.items():
        data = series(n)
        hypo = {"claim_type": "directional"}

        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            run_hypothesis(hypo, data)
            times.append((time.perf_counter() - t0) * 1_000)

        mean_ms = sum(times) / len(times)
        rows.append((label, n, mean_ms, min(times), max(times)))
    return rows


def peak_memory(test_name, data_fn, n=500, fixed_data=None):
    data = fixed_data if fixed_data is not None else data_fn(n)
    tracemalloc.start()
    run_test(test_name, data)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1_024  # KB


# ── Reporting ──────────────────────────────────────────────────────────────────

def header(title):
    print(f"\n{BOLD}── {title} {'─' * (52 - len(title))}{RESET}")

def print_rows(rows):
    print(f"  {DIM}{'size':<8} {'n':>6}   {'mean':>8}   {'min':>8}   {'max':>8}{RESET}")
    for label, n, mean_ms, min_ms, max_ms in rows:
        n_str = f"{n:,}" if n is not None else "—"
        bar   = "█" * max(1, int(mean_ms * 4))
        print(f"  {label:<8} {n_str:>6}   "
              f"{CYAN}{mean_ms:>6.3f} ms{RESET}   "
              f"{DIM}{min_ms:>6.3f} ms   {max_ms:>6.3f} ms{RESET}   "
              f"{GREEN}{bar}{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'═' * 56}")
    print(f"  Lighthouse 4 — Performance Benchmark")
    print(f"  {REPS} reps / {REPS_SLOW} reps (slow@large)   |   sizes: {list(SIZES.keys())}")
    print(f"{'═' * 56}{RESET}")

    # DIRECTIONAL
    header("DIRECTIONAL")
    print(f"  {DIM}mann_kendall_test{RESET}")
    print_rows(bench("mann_kendall_test", series, SIZES))
    print(f"\n  {DIM}linear_regression_test{RESET}")
    print_rows(bench("linear_regression_test", series, SIZES))

    # COMPARATIVE
    header("COMPARATIVE")
    print(f"  {DIM}t_test{RESET}")
    print_rows(bench("t_test", two_groups, SIZES))
    print(f"\n  {DIM}mann_whitney_test{RESET}")
    print_rows(bench("mann_whitney_test", two_groups, SIZES))
    print(f"\n  {DIM}anova_test{RESET}")
    print_rows(bench("anova_test", three_groups, SIZES))
    print(f"\n  {DIM}kruskal_wallis_test{RESET}")
    print_rows(bench("kruskal_wallis_test", three_groups, SIZES))
    print(f"\n  {DIM}tukey_hsd{RESET}")
    print_rows(bench("tukey_hsd", three_groups, SIZES))
    print(f"\n  {DIM}dunns_test{RESET}")
    print_rows(bench("dunns_test", three_groups, SIZES))

    # CAUSAL
    header("CAUSAL")
    print(f"  {DIM}difference_in_differences{RESET}")
    print_rows(bench("difference_in_differences", did_data, SIZES))
    print(f"\n  {DIM}interrupted_time_series{RESET}")
    print_rows(bench("interrupted_time_series", its_data, SIZES))
    print(f"\n  {DIM}detect_change_point  (ruptures — slowest test){RESET}")
    print_rows(bench("detect_change_point", cp_data, SIZES))

    # DISTRIBUTIONAL
    header("DISTRIBUTIONAL")
    print(f"  {DIM}z_test  (fixed input — does not scale with N){RESET}")
    print_rows(bench("z_test", None, fixed_data=FIXED_Z))
    print(f"\n  {DIM}chi_square_test{RESET}")
    print_rows(bench("chi_square_test", chi_data, SIZES))
    print(f"\n  {DIM}ks_test{RESET}")
    print_rows(bench("ks_test", ks_data, SIZES))

    # FULL PIPELINE
    header("FULL PIPELINE  (parse → select → run,  directional / mann_kendall)")
    print_rows(bench_pipeline(SIZES))

    # MEMORY
    header("PEAK MEMORY  (n=500 per test)")
    tests_mem = [
        ("mann_kendall_test",          series,       None),
        ("linear_regression_test",     series,       None),
        ("t_test",                     two_groups,   None),
        ("mann_whitney_test",          two_groups,   None),
        ("anova_test",                 three_groups, None),
        ("kruskal_wallis_test",        three_groups, None),
        ("difference_in_differences",  did_data,     None),
        ("interrupted_time_series",    its_data,     None),
        ("detect_change_point",        cp_data,      None),
        ("z_test",                     None,         FIXED_Z),
        ("chi_square_test",            chi_data,     None),
        ("ks_test",                    ks_data,      None),
    ]
    print(f"  {DIM}{'test':<34} {'peak KB':>8}{RESET}")
    for name, fn, fixed in tests_mem:
        kb = peak_memory(name, fn, fixed_data=fixed)
        bar = "█" * max(1, int(kb / 5))
        print(f"  {name:<34} {CYAN}{kb:>6.1f} KB{RESET}   {GREEN}{bar}{RESET}")

    # FOOTER
    print(f"\n{BOLD}{'═' * 56}{RESET}")
    print(f"  {YELLOW}All times are wall-clock (single-threaded).{RESET}")
    print(f"  {YELLOW}Memory is tracemalloc peak — excludes Python baseline.{RESET}")
    print(f"{BOLD}{'═' * 56}{RESET}\n")


if __name__ == "__main__":
    main()
