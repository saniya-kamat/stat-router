# PROTOTYPE README
*Built as a self-directed side project during my internship at Incedo, exploring hypothesis-testing automation independent of the production system.*

This is an automated hypothesis-testing prototype for business data that turns a statement like “churn is going up” or “revenue changed after the launch” into a statistical test, and then returns a verdict based on the underlying data and a p-value threshold. In plain language, a statistically supported statement means the system looked at the evidence in the data and found enough signal to reject the “no effect / no change / no trend” baseline with a chosen confidence level; otherwise, it fails to reject the null hypothesis.

To be clear, this prototype is a rule-based statistical routing engine, not an AI/ML system. The “intelligence” is in how it inspects your data and picks the right test — there is no model making predictions anywhere in the pipeline.

## Project Overview
Lighthouse 4 is designed to reduce the manual work of choosing the right statistical test for business questions. Instead of asking analysts to decide between trend tests, group-comparison tests, causal-style tests, and distribution tests, the system routes a hypothesis and its data into a test family automatically and returns an interpretable result. The prototype also supports a lightweight UI and sample datasets that demonstrate the main business cases.

## Routing Logic
The system starts by asking the user what claim they'd like to test and then routing to one of four hypothesis testing families based on their answer: Directional, Comparative, Causal, or Distributional. That claim type determines both the test-selection logic and the expected input shape, which is how the prototype avoids forcing every problem into the same statistical template. Once it knows the claim type, the system doesn't just pick one fixed test — it **inspects the actual data** to choose the most appropriate test for your exact scenario. 

### How Test Selection Works
The clever part of Lighthouse 4 is the selection step, which runs two real statistical checks on your data before committing to a test:

- **Normality (Shapiro–Wilk).** For group comparisons, the system tests whether each group looks normally distributed. If everything looks normal it uses the parametric test (t-test / ANOVA); otherwise it falls back to the non-parametric equivalent (Mann–Whitney / Kruskal–Wallis). To stay sensible at the extremes, it treats very small samples (n < 3) and very large ones (n > 5000) as “normal enough” and relies on the Central Limit Theorem rather than over-rejecting.
- **Autocorrelation (lag-1, Bartlett bound).** For trend detection on a time series, the system checks whether the series is autocorrelated. If it is, it prewhitens the data first so the autocorrelation doesn't artificially inflate the trend's significance.

Because the routing is driven by the data itself, the same claim type can legitimately resolve to different tests depending on what your numbers actually look like.

## Claim Types
### Directional
Directional claims answer questions like “Is this metric trending up or down over time?” or “Is the slope significantly different from zero?” The prototype expects a numeric time series. If you ask for a slope, it uses `linear_regression_test`; otherwise it uses `mann_kendall_test` for trend detection, prewhitening first if the series is autocorrelated.

### Comparative
Comparative claims answer questions like “Is Segment A performing better than Segment B?” or “Do three regions have different means?” The expected input is a dictionary with `groups`, where each group is an array of numeric values. Two groups map to `t_test` or `mann_whitney_test`; three or more groups map to `anova_test` or `kruskal_wallis_test` — with the parametric-vs-non-parametric choice decided by the normality check described above.

When an ANOVA or Kruskal–Wallis comes back significant, you know *that* the groups differ but not *which* pairs differ. For that, two post-hoc tests are available through the debug endpoint: `tukey_hsd` (the follow-up to ANOVA) and `dunns_test` (the follow-up to Kruskal–Wallis, with Bonferroni correction). These aren't auto-selected — you run them explicitly when you want the pairwise breakdown.

### Causal
Causal-style claims answer questions like “Did the intervention change the metric?” or “Is there a structural break?” The prototype currently expects either a treatment/control before-after structure for `difference_in_differences`, or a single time series for `interrupted_time_series` and `detect_change_point` (which uses the PELT change-point algorithm). Set `mode: "change_point"` on the data to ask specifically for break detection.

### Distributional
Distributional claims answer questions like “Is the current value abnormal versus baseline?” or “Do these two samples come from the same distribution?” The prototype expects either a baseline comparison object for `z_test`, categorical counts for `chi_square_test`, or two numeric samples for `ks_test` (selected with `mode: "distribution"`).

## Sample Datasets
### churn_trend.csv
This file contains 24 monthly churn-rate values with an upward linear trend plus small Gaussian noise. It is built to support directional testing, and `mann_kendall_test` or `linear_regression_test` should both show a strong positive trend; p < 0.001 is a reasonable expectation.

### segment_comparison.csv
This file contains 200 Northeast values and 200 Southwest values with different means and reasonable spread. It is built for comparative tests, and both `t_test` and `mann_whitney_test` should detect a statistically significant difference between segments.

### revenue_its.csv
This file contains 24 monthly revenue values with a clear level shift after month 12. It is built for causal-style testing, and `interrupted_time_series` should show a significant pre/post difference; `detect_change_point` should identify a breakpoint near the intervention.

### conversion_anomaly.csv
This file contains a single current conversion rate plus a historical mean and standard deviation. It is built for `z_test`, and the value 0.042 versus baseline 0.08 with standard deviation 0.01 should produce a very small p-value and a reject-null result.

## Running the Prototype
Install the dependencies and start the Flask app:

```bash
pip install -r requirements.txt
python main.py
```

The server runs on `http://127.0.0.1:5000` by default (set the `PORT` environment variable to override). Open that address in a browser for the lightweight UI, or call the API directly.

## Flow Diagram
`flow_diagram.drawio` contains a full visual map of this pipeline — from the incoming request through validation, parsing, claim-type routing, the per-branch test-selection decisions, execution, and interpretation. Open it with the Draw.io Integration extension in VS Code or at [app.diagrams.net](https://app.diagrams.net).