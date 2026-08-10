"""
Each test uses a dataset where the correct answer is mathematically
certain, so you can verify the backend returns the right result before
the presentation.
 
Pass = function returns correct significance verdict AND key statistics
       are in the expected direction/range.
"""
 
import sys
import numpy as np
from app.hypothesis_engine import run_test
 
# ── Color helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
 
passed = 0
failed = 0
warned = 0
 
 
def ok(name, note=""):
    global passed
    passed += 1
    print(f"  {GREEN}✓ PASS{RESET}  {name}" + (f"  [{note}]" if note else ""))
 
 
def fail(name, reason):
    global failed
    failed += 1
    print(f"  {RED}✗ FAIL{RESET}  {name}  ← {reason}")
 
 
def warn(name, reason):
    global warned
    warned += 1
    print(f"  {YELLOW}⚠ WARN{RESET}  {name}  ← {reason}")
 
 
def section(title):
    print(f"\n{BOLD}── {title} {'─' * (52 - len(title))}{RESET}")
 
 
# DIRECTIONAL TESTS
section("DIRECTIONAL")
 
# Mann–Kendall: clear upward trend → SIGNIFICANT
np.random.seed(0)
trend_up = list(np.linspace(10, 50, 30) + np.random.normal(0, 1, 30))
r = run_test("mann_kendall_test", trend_up)
if r["significant"] and r["trend"] == "increasing":
    ok("mann_kendall — upward trend", f"p={r['p_value']:.4f}, trend={r['trend']}")
else:
    fail("mann_kendall — upward trend", f"expected significant+increasing, got {r}")
 
# Mann–Kendall: flat noise → NOT significant
flat = list(np.random.normal(50, 1, 30))
r = run_test("mann_kendall_test", flat)
if not r["significant"]:
    ok("mann_kendall — flat series", f"p={r['p_value']:.4f}")
else:
    fail("mann_kendall — flat series", f"expected not-significant, got p={r['p_value']:.4f}")
 
# Linear Regression: strong positive slope → SIGNIFICANT
x_vals = list(np.linspace(1, 100, 50) + np.random.normal(0, 3, 50))
r = run_test("linear_regression_test", x_vals)
if r["significant"] and r["slope"] > 0:
    ok("linear_regression — positive slope", f"slope={r['slope']:.3f}, p={r['p_value']:.4f}")
else:
    fail("linear_regression — positive slope", f"got {r}")

# Linear Regression: flat line → MUST NOT detect trend
flat = [10.0] * 50
r = run_test("linear_regression_test", flat)

if r["slope"] == 0.0 and r["p_value"] > 0.99:
    ok("linear_regression — no trend", f"p={r['p_value']:.4f}")
else:
    fail(
        "linear_regression — no trend",
        f"expected flat line, got slope={r['slope']}, p={r['p_value']:.4f}"
    )


# COMPARATIVE TESTS
section("COMPARATIVE")
 
np.random.seed(42)
# T-test: means 100 vs 130, tight std → SIGNIFICANT
group_a = list(np.random.normal(100, 5, 40))
group_b = list(np.random.normal(130, 5, 40))
r = run_test("t_test", {"groups": [group_a, group_b]})
if r["significant"] and r["statistic"] < 0:  # a < b → negative t
    ok("t_test — clearly different means", f"t={r['statistic']:.3f}, p={r['p_value']:.4e}")
else:
    fail("t_test — clearly different means", f"got {r}")
 
# T-test: same distribution → NOT significant
group_c = list(np.random.normal(100, 5, 40))
group_d = list(np.random.normal(100, 5, 40))
r = run_test("t_test", {"groups": [group_c, group_d]})
if not r["significant"]:
    ok("t_test — same distribution", f"p={r['p_value']:.4f}")
else:
    warn("t_test — same distribution", f"got significant (p={r['p_value']:.4f}) — check random seed; can occur by chance")
 
# Mann–Whitney: heavily skewed, different medians → SIGNIFICANT
skewed_a = list(np.random.exponential(scale=2, size=50))
skewed_b = list(np.random.exponential(scale=8, size=50))
r = run_test("mann_whitney_test", {"groups": [skewed_a, skewed_b]})
if r["significant"]:
    ok("mann_whitney — different medians (skewed)", f"p={r['p_value']:.4e}")
else:
    fail("mann_whitney — different medians (skewed)", f"expected significant, got p={r['p_value']:.4f}")
 
# ANOVA: 3 clearly different groups → SIGNIFICANT
g1 = list(np.random.normal(10, 1, 30))
g2 = list(np.random.normal(20, 1, 30))
g3 = list(np.random.normal(30, 1, 30))
r = run_test("anova_test", {"groups": [g1, g2, g3]})
if r["significant"] and r["f_statistic"] > 1:
    ok("anova — 3 distinct groups", f"F={r['f_statistic']:.1f}, p={r['p_value']:.4e}")
else:
    fail("anova — 3 distinct groups", f"got {r}")
 
# Kruskal–Wallis: same as above but non-parametric → SIGNIFICANT
r = run_test("kruskal_wallis_test", {"groups": [g1, g2, g3]})
if r["significant"]:
    ok("kruskal_wallis — 3 distinct groups", f"H={r['h_statistic']:.1f}, p={r['p_value']:.4e}")
else:
    fail("kruskal_wallis — 3 distinct groups", f"got {r}")
 
 
# CAUSAL TESTS
section("CAUSAL")
 
np.random.seed(7)
 
# Diff-in-Differences: treatment jumps +12k, control flat → DiD ≈ +12k
tb = list(np.random.normal(50000, 1000, 12))
ta = list(np.random.normal(62000, 1000, 12))
cb = list(np.random.normal(30000, 1000, 12))
ca = list(np.random.normal(30000, 1000, 12))   # control does NOT change
r = run_test("difference_in_differences", {
    "treatment_before": tb, "treatment_after": ta,
    "control_before": cb,   "control_after": ca
})
did = r["did_estimate"]
if 10000 < did < 14000:
    ok("diff_in_differences — +12k treatment jump", f"DiD estimate={did:,.0f}")
else:
    fail("diff_in_differences — +12k treatment jump", f"expected ~12000, got {did:,.0f}")
 
# NOTE: DiD returns no p-value by design (prototype).
# If you need significance, add a paired t-test on the group deltas.
 
# Interrupted Time Series: clear level shift at index 10 → SIGNIFICANT
pre_rev  = list(np.random.normal(50000, 2000, 10))
post_rev = list(np.random.normal(65000, 2000, 10))
series   = pre_rev + post_rev
r = run_test("interrupted_time_series", {"series": series, "intervention_index": 10})
if r["significant"] and r["post_mean"] > r["pre_mean"]:
    ok("interrupted_time_series — level shift at t=10",
       f"pre={r['pre_mean']:,.0f}  post={r['post_mean']:,.0f}  p={r['p_value']:.4f}")
else:
    fail("interrupted_time_series — level shift", f"got {r}")
 
# Interrupted Time Series: no shift → NOT significant
flat_series = list(np.random.normal(50000, 1500, 20))
r = run_test("interrupted_time_series", {"series": flat_series, "intervention_index": 10})
if not r["significant"]:
    ok("interrupted_time_series — no shift", f"p={r['p_value']:.4f}")
else:
    warn("interrupted_time_series — no shift",
         f"significant at p={r['p_value']:.4f} (possible false positive with this seed)")
 
# Change Point Detection: one clear break at index 10
cp_series = list(np.random.normal(0, 1, 10)) + list(np.random.normal(20, 1, 10))
r = run_test("detect_change_point", {"series": cp_series})
detected = r["change_point"]
if detected is not None and 8 <= detected <= 12:
    ok("detect_change_point — break at index 10", f"detected at index {detected}")
elif detected is None:
    fail("detect_change_point — break at index 10",
         "no change point found — this is the pen=3 scale bug (see note below)")
else:
    warn("detect_change_point — break at index 10",
         f"detected at {detected} (expected ~10). pen=3 may need tuning for your data scale.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTIONAL TESTS
# ─────────────────────────────────────────────────────────────────────────────
section("DISTRIBUTIONAL")
 
# Z-test: conversion rate 0.042 vs historical 0.08 ± 0.01 → z ≈ -3.8, SIGNIFICANT
r = run_test("z_test", {"current_value": 0.042, "historical_mean": 0.08, "historical_std": 0.01})
if r["significant"] and r["z_score"] < -3:
    ok("z_test — anomalously low conversion rate",
       f"z={r['z_score']:.3f}, p={r['p_value']:.4e}")
else:
    fail("z_test — anomalously low conversion rate", f"got {r}")
 
# Z-test: value within 1 std → NOT significant
r = run_test("z_test", {"current_value": 0.081, "historical_mean": 0.08, "historical_std": 0.01})
if not r["significant"]:
    ok("z_test — value within normal range", f"z={r['z_score']:.3f}, p={r['p_value']:.4f}")
else:
    fail("z_test — value within normal range", f"expected not-significant, got p={r['p_value']:.4f}")
 
# Chi-square: observed [50, 30, 20] vs expected [33, 33, 34] → very different → SIGNIFICANT
r = run_test("chi_square_test", {"observed": [50, 30, 20], "expected": [33, 33, 34]})
if r["significant"]:
    ok("chi_square — unequal distribution vs uniform",
       f"χ²={r['chi_square_statistic']:.3f}, p={r['p_value']:.4f}")
else:
    fail("chi_square — unequal distribution vs uniform", f"got {r}")
 
# Chi-square: observed ≈ expected → NOT significant
r = run_test("chi_square_test", {"observed": [33, 34, 33], "expected": [33, 33, 34]})
if not r["significant"]:
    ok("chi_square — observed ≈ expected", f"p={r['p_value']:.4f}")
else:
    fail("chi_square — observed ≈ expected", f"expected not-significant, got p={r['p_value']:.4f}")
 
# KS test: two clearly different distributions → SIGNIFICANT
np.random.seed(1)
dist_a = list(np.random.normal(0,  1, 100))
dist_b = list(np.random.normal(5,  1, 100))
r = run_test("ks_test", {"sample_a": dist_a, "sample_b": dist_b})
if r["significant"] and r["ks_statistic"] > 0.5:
    ok("ks_test — different distributions (μ=0 vs μ=5)",
       f"KS={r['ks_statistic']:.3f}, p={r['p_value']:.4e}")
else:
    fail("ks_test — different distributions", f"got {r}")
 
# KS test: same distribution → NOT significant
dist_c = list(np.random.normal(0, 1, 100))
dist_d = list(np.random.normal(0, 1, 100))
r = run_test("ks_test", {"sample_a": dist_c, "sample_b": dist_d})
if not r["significant"]:
    ok("ks_test — same distribution", f"KS={r['ks_statistic']:.3f}, p={r['p_value']:.4f}")
else:
    warn("ks_test — same distribution",
         f"significant at p={r['p_value']:.4f} (possible false positive — can occur by chance)")
 
 
# SUMMARY
total = passed + failed + warned
print(f"\n{'═' * 56}")
print(f"  {BOLD}Results:{RESET}  "
      f"{GREEN}{passed} passed{RESET}  "
      f"{RED}{failed} failed{RESET}  "
      f"{YELLOW}{warned} warned{RESET}  "
      f"/ {total} total")
print(f"{'═' * 56}")
 
if failed == 0 and warned == 0:
    print(f"\n  {GREEN}{BOLD}All tests passed. Backend is correct.{RESET}\n")
elif failed == 0:
    print(f"\n  {YELLOW}No failures. Review warnings above — "
          f"most are expected statistical variance.{RESET}\n")
else:
    print(f"\n  {RED}Fix failures before the presentation.{RESET}\n")