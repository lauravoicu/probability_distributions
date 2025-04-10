import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import t

from graphics_style import apply_graphics_style

apply_graphics_style()

# Set random seed for reproducibility
np.random.seed(42)

# Simulate loss distributions
loss_without_mitigation = np.random.normal(loc=5_000_000, scale=2_000_000, size=10000)
loss_with_mitigation = np.random.normal(loc=4_800_000, scale=1_800_000, size=10000)


# Two-sample t-test (Welch’s, no equal variance assumption)
t_stat, p_value = ttest_ind(loss_without_mitigation, loss_with_mitigation, equal_var=False)

# Print results
print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("The difference is statistically significant—likely not just noise.")
else:
    print("The difference may be due to random variation.")

# Visualize the distributions
plt.figure(figsize=(10, 6))
plt.hist(loss_without_mitigation, bins=50, alpha=0.5, label="Without Control", color="grey")
plt.hist(loss_with_mitigation, bins=50, alpha=0.5, label="With Control", color="darkred")
plt.axvline(x=1_200_000, color="grey", linestyle="--", label="Mean Without Control")
plt.axvline(x=950_000, color="darkred", linestyle="--", label="Mean With Control")
plt.title("Loss Distributions: With vs. Without Endpoint Protection")
plt.xlabel("Annual Loss ($)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('power_analysis_p_normal.png')
plt.close()


mean_diff = np.mean(loss_without_mitigation) - np.mean(loss_with_mitigation)
pooled_std = np.sqrt((np.std(loss_without_mitigation)**2 + np.std(loss_with_mitigation)**2) / 2)
cohens_d = mean_diff / pooled_std

n1, n2 = len(loss_without_mitigation), len(loss_with_mitigation)
s1, s2 = np.std(loss_without_mitigation, ddof=1), np.std(loss_with_mitigation, ddof=1)
se = np.sqrt(s1**2/n1 + s2**2/n2)
df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
ci = t.interval(0.95, df, loc=mean_diff, scale=se)

print(f"Mean Reduction: ${mean_diff:,.0f} ({mean_diff/5_000_000:.1%} of $5M)")
print(f"Cohen’s d: {cohens_d:.2f} (small if < 0.2)")
print(f"95% CI: ${ci[0]:,.0f} to ${ci[1]:,.0f}")

# Visualize Effect Size and CI
plt.figure(figsize=(8, 4))
plt.errorbar(x=0, y=mean_diff, yerr=[mean_diff - ci[0]], fmt='o', capsize=5, label="Mean Reduction with 95% CI", color="green")
plt.axhline(y=0, color="gray", linestyle="--")
plt.xlim(-1, 1)
plt.xticks([])
plt.ylabel("Reduction in ALE ($)")
plt.title("Effect Size and Confidence Interval of Risk Reduction")
plt.legend()
plt.savefig('power_analysis_ci_normal.png')  # Save for blog use
plt.close()

# Simulate skewed loss distributions
loss_without_mitigation_skewed = np.random.lognormal(mean=15.4, sigma=0.5, size=10000)
loss_with_mitigation_skewed = np.random.lognormal(mean=15.35, sigma=0.45, size=10000)

# Mann-Whitney U test
u_stat, p_value_skewed = mannwhitneyu(loss_without_mitigation_skewed, loss_with_mitigation_skewed)

# Print results
print(f"U-statistic: {u_stat:.2f}")
print(f"P-value (skewed data): {p_value_skewed:.4f}")
if p_value_skewed < 0.05:
    print("The difference is statistically significant, even with skewed data.")
else:
    print("The difference may be due to random variation.")

# Visualize skewed distributions
plt.figure(figsize=(10, 6))
plt.hist(loss_without_mitigation_skewed, bins=50, alpha=0.5, label="Without Control (Skewed)", color="grey")
plt.hist(loss_with_mitigation_skewed, bins=50, alpha=0.5, label="With Control (Skewed)", color="darkred")
plt.axvline(x=1_200_000, color="grey", linestyle="--", label="Mean Without Control")
plt.axvline(x=950_000, color="darkred", linestyle="--", label="Mean With Control")
plt.title("Skewed Loss Distributions: With vs. Without Endpoint Protection")
plt.xlabel("Annual Loss ($)")
plt.ylabel("Frequency")
plt.legend()
plt.savefig('power_analysis_p_lognormal.png')
plt.close()

mean_diff_skewed = np.mean(loss_without_mitigation_skewed) - np.mean(loss_with_mitigation_skewed)
s1_skewed, s2_skewed = np.std(loss_without_mitigation_skewed, ddof=1), np.std(loss_with_mitigation_skewed, ddof=1)
se_skewed = np.sqrt(s1_skewed**2/n1 + s2_skewed**2/n2)
df_skewed = (s1_skewed**2/n1 + s2_skewed**2/n2)**2 / ((s1_skewed**2/n1)**2/(n1-1) + (s2_skewed**2/n2)**2/(n2-1))
ci_skewed = t.interval(0.95, df_skewed, loc=mean_diff_skewed, scale=se_skewed)

print(f"Mean Reduction (skewed): ${mean_diff_skewed:,.0f}")
print(f"95% CI (skewed): ${ci_skewed[0]:,.0f} to ${ci_skewed[1]:,.0f}")

# Visualize Effect Size and CI for Skewed Data
plt.figure(figsize=(8, 4))
plt.errorbar(x=0, y=mean_diff_skewed, yerr=[mean_diff_skewed - ci_skewed[0]], fmt='o', capsize=5, label="Mean Reduction with 95% CI (Skewed)", color="purple")
plt.axhline(y=0, color="gray", linestyle="--")
plt.xlim(-1, 1)
plt.xticks([])
plt.ylabel("Reduction in ALE ($)")
plt.title("Effect Size and Confidence Interval (Skewed Data)")
plt.legend()
plt.savefig('power_analysis_ci_lognormal.png')
plt.close()