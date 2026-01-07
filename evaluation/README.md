# 📊 Performance Evaluation Guide
This guide explains how to evaluate your controllers and generate comparative statistical reports.

## 1. Single Model Evaluation
Run the evaluation script for each controller individually. This generates the necessary summary CSVs and raw episode data files required for the statistical analysis.

### A) Evaluate the Baseline (Lyapunov)
By default, running without a model path evaluates the analytical Lyapunov controller.

```console
python ./scripts/run_policy_evaluation.py --mode single
```
### B) Evaluate RL Policies
To evaluate a trained model, provide the path to the policy file and a custom name for the report.

```console
python ./scripts/run_policy_evaluation.py --mode single --model training/v1_no_baseline/policy_model.pth --name v1_no_baseline
```


## 2. Multi-Controller Comparison
Once you have evaluated at least two controllers, run the comparison mode. This will:

- Generate a Leaderboard: A table comparing Mean ± Std Dev for all metrics.

- Statistical Analysis: Perform Wilcoxon Signed-Rank tests and calculate Cliff's Delta effect sizes (comparing your RL models against the baseline).

- Distribution Plots: Save comparative boxplots for each metric in evaluation/output/boxplots/.

```console
python ./scripts/run_policy_evaluation.py --mode compare
```


## 3. Robustness Testing (Optional)
Max Curve Test (Radius Sweep)
To test how the controller handles different path curvatures, run a radius sweep. This will test the model across a range of radii and plot the success/error curves.

```console
python ./scripts/run_policy_evaluation.py --mode sweep --model training/v1_no_baseline/policy_model.pth
```

## Output Structure
All results are stored in the evaluation/output/ directory:

- TT_policy_comparison.csv: Summary of all single evaluations runs.

- raw_episode_data/: Individual episode results used for statistical testing.

- boxplots/: Visual distribution of metrics (CTE, Energy, Smoothness, etc.).

- sweep_scatter_policy_model.png: Visual result of the radius sweep.

