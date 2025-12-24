# Basic Statistics

Statistics is the foundation of machine learning. Before any model can learn patterns, we need to understand the data itself—how to collect it properly, describe it meaningfully, and make valid inferences from samples to populations.

## Sampling and Measurement

When we collect data, we're measuring characteristics (called **variables**) of our subjects:

**Types of Variables**:
- **Quantitative** (Numerical): Things we can measure with numbers (height, temperature, income)
- **Categorical** (Qualitative): Things we describe with categories (color, species, disease status)

**Measurement Scales**—different types require different statistical treatments:
- **Interval Scale** (Quantitative): Differences are meaningful, but no true zero (temperature in Celsius—0°C doesn't mean "no temperature")
- **Nominal Scale** (Qualitative): Unordered categories (eye color, blood type)—can only check equality
- **Ordinal Scale** (Qualitative): Ordered categories (education level, Likert scale ratings)—can rank but can't quantify differences

**The Sampling Problem**: Any statistic we compute from a sample will vary from sample to sample. This variation comes from two sources:
- **Sampling Error**: Unavoidable random variation from using a sample instead of the whole population
- **Sampling Bias**: Systematic errors in how we collected the sample
    - *Selection Bias*: Some population members more likely to be included (surveying only online users)
    - *Response Bias*: People don't answer truthfully (socially desirable responses)
    - *Non-Response Bias*: Certain types of people don't respond (busy people skip surveys)

**Sampling Methods**:
- **Simple Random Sampling**: Each data point has equal probability of being selected—the gold standard but sometimes impractical
- **Stratified Random Sampling**: Divide population into meaningful subgroups (strata), then sample from each
    - Ensures representation from all important subgroups
    - Often produces lower variance estimates than simple random sampling
    - Example: Stratifying by age groups when studying health outcomes
- **Cluster Sampling**: Divide into clusters (often geographic), randomly select clusters, sample within them
    - More practical for large geographic areas
    - Example: Randomly selecting cities, then sampling households within those cities
- **Multi-Stage Sampling**: Combination of sampling methods at different stages

## Descriptive Statistics

Descriptive statistics summarize what the data looks like before we make inferences.

**Measures of Central Tendency**:
- **Mean** ($\bar{x}$): The arithmetic average—sensitive to outliers
- **Median**: The middle value when sorted—robust to outliers
- **Mode**: The most frequent value—useful for categorical data

**Shape of Distribution**—understanding where mean and median fall:
- **Symmetric**: Mean coincides with median (normal distribution is symmetric)
- **Left Skewed** (negatively skewed): Long left tail → Mean < Median
    - Example: Age at retirement (most retire around 65, some retire very young)
- **Right Skewed** (positively skewed): Long right tail → Mean > Median
    - Example: Income distribution (most earn moderate amounts, few earn millions)
- **Key insight**: The mean is "pulled" toward the long tail

**Standard Deviation**—measures spread around the mean:
- Deviation = how far each observation is from the mean
- $s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N-1}}$
- Why $N-1$? This is "Bessel's correction"—using $N$ would underestimate the true population variance

**The 68-95-99.7 Rule** (Empirical Rule) for normally distributed data:
- ~68% of data falls within 1 standard deviation of the mean
- ~95% of data falls within 2 standard deviations
- ~99.7% of data falls within 3 standard deviations (the "3-sigma rule")

**Interquartile Range (IQR)**:
- IQR = 75th percentile − 25th percentile (the middle 50% of data)
- Common outlier rule: Points beyond $1.5 \times \text{IQR}$ from Q1 or Q3 are potential outliers

## Probability

Probability quantifies uncertainty. Here's how we formalize it:

**Expected Value** (what you'd "expect" on average):
- $E(X) = \sum_i x_i \times p(X=x_i)$
- This is the first moment about the origin—the "center of mass" of the distribution

**Variance** (how spread out the values are):
- $V(X) = E(X^2) - (E(X))^2 = E[(X - \mu)^2]$
- This is the second moment about the mean

**Z-score** (standardization):
- $z = \frac{y - \mu}{\sigma}$
- Transforms any normal distribution to the **Standard Normal Distribution** $\sim N(0,1)$
- Interpretation: "How many standard deviations away from the mean?"

**Covariance and Correlation**:
- Covariance: $Cov(X,Y) = E[(X - \mu_x)(Y - \mu_y)]$—direction of linear relationship
- Correlation: $\rho = \frac{Cov(X,Y)}{\sigma_x \sigma_y} = E(z_x z_y)$—standardized covariance, ranges from -1 to +1

**The Central Limit Theorem (CLT)**—one of the most important results in statistics:
- Sample means follow an approximately normal distribution regardless of the population distribution
- $\bar{X} \sim N(\mu, \frac{\sigma}{\sqrt{N}})$
- **Standard Error** = $\frac{\sigma}{\sqrt{N}}$—the standard deviation of the sampling distribution
- Key insight: Standard error decreases with $\sqrt{N}$, not $N$ (need 4x samples to halve error)

**Example: Exit Poll Survey**
- Binary outcome (vote A or B), assume $p = 0.5$, sample size $N = 1800$
- Standard deviation: $\sqrt{p(1-p)} = 0.5$
- Standard error: $\frac{0.5}{\sqrt{1800}} \approx 0.012$
- 99% CI: $0.5 \pm 3 \times 0.012 \approx (0.47, 0.53)$

**Example: Income Survey**
- Population: $\sim N(380, 80^2)$ (mean \$380K, SD \$80K), sample size $N = 100$
- Question: What's $P(\bar{y} \geq 400)$?
- Standard error: $\frac{80}{\sqrt{100}} = 8$
- $z = \frac{400 - 380}{8} = 2.5$
- $P(Z \geq 2.5) < 0.006$ (very unlikely to see sample mean ≥ 400 by chance)

## Confidence Interval

Confidence intervals quantify our uncertainty about parameter estimates.

**Point Estimate**: A single number as our best guess (e.g., sample mean $\bar{x}$ for population mean $\mu$)

**Properties of Good Estimators**:
- **Unbiased**: $E(\bar{X}) = \mu$ (on average, the estimator equals the true value)
- **Efficient/Consistent**: $N \to \infty \implies V(\bar{X}) \to 0$ (variance shrinks as sample grows)

**Interval Estimate**: A range of plausible values
- CI = Point Estimate ± Margin of Error
- **Confidence Level**: The probability (e.g., 95%) that the procedure produces an interval containing the true parameter
- Important: A 95% CI doesn't mean "95% probability the parameter is in this interval"—the parameter is fixed, not random!

**CI for Proportions**:
- Point estimate: $\hat{\pi}$
- Standard error: $\sqrt{\frac{\hat{\pi}(1-\hat{\pi})}{N}}$
- 95% CI: $\hat{\pi} \pm z_{0.025} \times se$ where $z_{0.025} \approx 1.96$

**CI for Means**:
- Point estimate: $\bar{X}$
- When population variance is unknown, use sample variance and **t-distribution** instead of z-distribution
- The t-distribution has heavier tails—accounts for extra uncertainty from estimating variance
- As $N \to \infty$, t-distribution approaches normal distribution
- CI: $\bar{X} \pm t_{n-1,\alpha/2} \times \frac{s}{\sqrt{N}}$

**Sample Size Determination**:
- For proportions: $N = \frac{\pi(1-\pi) \times z^2}{M^2}$ where $M$ is desired margin of error
- For means: $N = \frac{\sigma^2 \times z^2}{M^2}$
- Note: Quadrupling sample size only halves the margin of error!

**Estimation Methods**:
- **Maximum Likelihood Estimation (MLE)**: Find parameter values that maximize the probability of observing the data we actually observed
- **Bootstrap**: Resample from observed data to estimate standard errors and CIs—no distributional assumptions needed

## Significance Test

Hypothesis testing provides a framework for making decisions based on data.

**Five Components of a Significance Test**:

1. **Assumptions**: What conditions must hold?
    - Type of data, randomization, population distribution, sample size

2. **Hypotheses**:
    - $H_0$ (Null): The "no effect" or "status quo" hypothesis
    - $H_1$ (Alternative): What we're testing for

3. **Test Statistic**: Quantifies how far the observed data falls from what $H_0$ predicts

4. **P-value**: Probability of observing data as extreme or more extreme than what we got, *assuming $H_0$ is true*
    - Small p-value → data is unlikely under $H_0$ → evidence against $H_0$
    - P-value is NOT the probability that $H_0$ is true!

5. **Conclusion**: "Reject $H_0$" or "Fail to reject $H_0$"
    - Note: We never "accept $H_0$"—absence of evidence is not evidence of absence

**Testing Proportions**:
- $H_0: \pi = \pi_0$ vs $H_1: \pi \neq \pi_0$
- Test statistic: $z = \frac{\hat{\pi} - \pi_0}{se}$ where $se = \sqrt{\frac{\pi_0(1-\pi_0)}{N}}$

**Testing Means**:
- $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$
- Test statistic: $t = \frac{\bar{X} - \mu_0}{s/\sqrt{N}}$
- For small samples, use exact binomial distribution

**One-Tailed vs Two-Tailed Tests**:
- One-tailed: Tests deviation in one direction only
- More powerful for detecting effects in that direction
- Risky: Can't detect effects in the opposite direction
- Use only with strong prior justification

**Types of Errors**:
- **Type I Error** (False Positive): Reject $H_0$ when it's actually true
    - Probability = significance level ($\alpha$, often 0.05)
- **Type II Error** (False Negative): Fail to reject $H_0$ when it's actually false
    - Probability denoted $\beta$
- **Power** = $1 - \beta$ = probability of correctly rejecting false $H_0$

**Trade-offs**:
- Decreasing Type I error increases Type II error
- The closer the true parameter is to $H_0$, the lower the power
- Larger samples → more power

**Important Warning**: Statistical significance ≠ practical significance. With large enough samples, tiny, meaningless differences become "statistically significant."

## Comparison of Groups

Most research involves comparing groups—treatments vs control, different populations, etc.

**Difference in Means (Independent Samples)**:
- Goal: Test if $\mu_1 - \mu_2 = 0$
- Estimate: $\bar{y}_1 - \bar{y}_2$
- Standard error: $se = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$
- CI: $(\bar{y}_1 - \bar{y}_2) \pm t \times se$
- Test statistic: $t = \frac{(\bar{y}_1 - \bar{y}_2) - 0}{se}$ with $df = n_1 + n_2 - 2$

**Equal Variance Assumption** (Pooled Variance):
- If we assume $\sigma_1 = \sigma_2$, we can pool the data to get a better variance estimate:
- $s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$
- $se = s_{pooled}\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$

**Difference in Proportions**:
- Standard error: $se = \sqrt{\frac{\hat{\pi}_1(1-\hat{\pi}_1)}{n_1} + \frac{\hat{\pi}_2(1-\hat{\pi}_2)}{n_2}}$
- For significance test, pool proportions under $H_0$:
- $\hat{\pi}_{pooled}$ and $se = \sqrt{\hat{\pi}(1-\hat{\pi})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}$

**Paired/Matched Samples**:
- Same subject measured at different times (before/after)
- Controls for between-subject variation
- Analyze the *differences* directly
- Test: $t = \frac{\bar{d} - 0}{s_d/\sqrt{n}}$
- Example: Blood pressure before and after treatment for the same patients

**Effect Size**: Standardized measure of the magnitude of difference
- Cohen's d: $d = \frac{\bar{y}_1 - \bar{y}_2}{s_{pooled}}$
- Interpretation: Small (~0.2), Medium (~0.5), Large (~0.8)

**McNemar Test** (for paired proportions):

| | Treatment Yes | Treatment No |
|--|---------------|--------------|
| **Control Yes** | $n_{11}$ | $n_{12}$ |
| **Control No** | $n_{21}$ | $n_{22}$ |

- Test statistic: $z = \frac{n_{12} - n_{21}}{\sqrt{n_{12} + n_{21}}}$
- Only the discordant pairs ($n_{12}$ and $n_{21}$) matter!

**Non-Parametric Alternatives** (when normality is violated):
- **Wilcoxon Signed-Rank Test**: For paired data—ranks the absolute differences
- **Mann-Whitney U Test**: For independent samples—compares ranks between groups

## Association between Categorical Variables

When both variables are categorical, we analyze their relationship through contingency tables.

**Statistical Independence**: Variables are independent if knowing one tells you nothing about the other. Mathematically: $P(A|B) = P(A)$

**Chi-Square Test**:
1. Calculate expected frequencies under independence:
   $f_e = \frac{\text{row total} \times \text{column total}}{\text{grand total}}$
2. Compare to observed frequencies:
   $\chi^2 = \sum \frac{(f_o - f_e)^2}{f_e}$
3. Degrees of freedom: $(r-1) \times (c-1)$

**Important**: $\chi^2$ tells you *if* there's an association, not *how strong* it is!

**Residual Analysis**—which cells drive the association?
- Standardized residual: $z = \frac{f_o - f_e}{\sqrt{f_e(1 - \text{row\%})(1 - \text{col\%})}}$
- $|z| > 2$ suggests the cell is significantly different from expected

**Odds Ratio** (2×2 tables):
- $\theta = \frac{n_{11} \times n_{22}}{n_{12} \times n_{21}}$
- Interpretation:
    - $\theta = 1$: No association (equal odds)
    - $\theta > 1$: Higher odds for row 1
    - $\theta < 1$: Lower odds for row 1
- Example: If odds ratio for smoking and lung cancer is 10, smokers have 10× the odds of lung cancer

**Ordinal Variables**—when categories have a natural order:
- **Concordant pairs**: Higher on X corresponds to higher on Y
- **Discordant pairs**: Higher on X corresponds to lower on Y
- **Gamma coefficient**: $\gamma = \frac{C - D}{C + D}$
    - Ranges from -1 to +1, like correlation for ordinal data

## P-value Interpretation and Common Misconceptions

Understanding what the p-value actually means is crucial:

**What p-value IS**:
- The probability of obtaining a test statistic at least as extreme as observed, *given that the null hypothesis is true*
- A measure of how compatible the data is with the null hypothesis

**What p-value is NOT**:
- ❌ The probability that the null hypothesis is true
- ❌ The probability that results occurred "by chance"
- ❌ The probability of making a Type I error (that's $\alpha$, which you set beforehand)

**Guidelines for Interpretation**:
- Small p-value (e.g., < 0.05): Data is unlikely under $H_0$—evidence against $H_0$
- Large p-value: Data is compatible with $H_0$—but doesn't prove $H_0$ is true
- Always report effect sizes alongside p-values
- Consider practical significance, not just statistical significance
