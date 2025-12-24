# Basic Statistics

## Sampling and Measurement

-   A characteristic that can be measured for each data point
    -   Quantitative: Numerical
    -   Categorical: Categories
-   Measurement Scales:
    -   Quantitative: Interval Scale
    -   Qualitative:
        -   Nominal Scale (Unordered)
        -   Ordinal Scale (Ordered)
-   Statistic varies from sample to sample drawn from the same distribution
    -   Sampling Bias + Sampling Error
-   Sampling Error
    -   Error that occurs on account of using a sample to calculate the population statistic
-   Sampling Bias
    -   Selection Bias
    -   Response Bias
    -   Non-Response Bias
-   Simple Random Sampling
    -   Each data point has equal probability of being selected
-   Stratified Random Sampling
    -   Divide population into strata and select random samples from each
    -   Ensures representation from all important subgroups
    -   Particularly useful when subgroups vary significantly in characteristics
    -   Often produces lower variance in estimates compared to simple random sampling
-   Cluster Sampling
    -   Divide population into clusters and select select random samples from each
-   Multi-Stage Sampling
    -   Combination of sampling methods

## Descriptive Statistics

-   Mean, Median, Mode
-   Shape of Distribution
    -   Symmetric around the central value
        -   Mean coincides with median
    -   Left Skewed: Left tail is longer
        -   Mean < Median
    -   Right Skewed: Right tail is longer
        -   Mean > Median
    -   For skewed distributions, mean lies closer to the long tail
-   Standard Deviation:
    -   Deviation is difference of observation from mean
    -   $s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N-1}}$
    -   Measures variability around mean
    -   For normally distributed data:
        -   Approximately 68% of data falls within 1 standard deviation of the mean
        -   Approximately 95% of data falls within 2 standard deviations of the mean
        -   Approximately 99.7% of data falls within 3 standard deviations of the mean (3-sigma rule)
-   IQR: Inter Quartile Range
    -   Difference between 75th and 25th percentile
    -   Outlier falls beyond 1.5 x IQR
-   Empirical Rule:
    -   For bell-shaped distributions - 68% volume is within 1 sdev and 95% volume within 2 sdev

## Probability

-   $E(X) = \sum_i x_i \times p(X=x_i)$
    -   First moment about origin
-   $V(X) = E(X^2) - (E(X))^2$
    -   Second moment about mean
-   $z = (y - \mu) / \sigma$
-   Standard Normal Distribution $\sim N(0,1)$
-   $Cov(X,Y) = E[(X - \mu_x)(Y - \mu_y)]$
-   Correlation $\rho = Cov(X,y) / \sigma_x \sigma_y = E(z_x z_y)$
-   Sampling Distribution: Probability distribution of the test statistic
-   Sample Mean
    -   Central Limit Theorem
    -   $\sim N(\mu, \sigma / \sqrt N)$
    -   Standard Error $\sigma / \sqrt N$
    -   Standard Deviation of Sampling Distribution
-   Case: Exit poll survey
    -   $\sim B(0.5)$ with sample size 1800
    -   Variance $\sqrt{p (1-p)}$ = 0.25
    -   Standard Error $\sigma / \sqrt N$ = 0.01
    -   99% CI: $0.5 \pm 3 * 0.01 \approx (0.47, 0.53)$
-   Case: Income Survey
    -   $\sim N(380, 80^2)$ with sample size 100
    -   $P(\bar y >= 400)$
    -   Standard Error $\sigma / \sqrt N$ = 8
    -   $z = (400 - 380) / 8 = 2.5$
    -   $P(Z >= z) < 0.006$

## Confidence Interval

-   Point Estimate: Single number representing the best guess for the parameter
-   Unbiased Estimator:
    -   $E(\bar X) = \mu$
    -   In expectation the estimator converges to the true population value
-   Efficient Estimator:
    -   $N \to \inf \implies V(\bar X) \to 0$
    -   The standard error approaches to zero as the sample size increases
-   Interval Estimate:
    -   Confidence Interval: Range of values that can hold the true parameter value
    -   Confidence Value: Probability with which true parameter value lies in CI
    -   Point Estimate $\pm$ Margin of Error
-   CI for Proportion
    -   Point Estimate $\hat \pi$
    -   Variance $\hat \sigma^2 = \hat \pi (1 - \hat \pi)$
    -   Standard Error: $\hat \sigma / \sqrt N = \sqrt{ \hat \pi (1 - \hat \pi) / N}$
    -   99% CI = $\hat \pi \pm (z_{0.01} \times se)$
    -   $(z_{0.01} \times se)$ is the margin of error
    -   Confidence Level increases the CI
    -   Sample Size decreases the CI
    -   Type 1 Error Propability: 1 - confidence level
-   CI for Mean
    -   Point Estimate $\hat \mu = \bar X$
    -   Variance $\hat \sigma^2 = \sum (X_i - \bar X)^2 / (N-1)$\
    -   Standard Error: $\hat \sigma / \sqrt N$
    -   True population variance is unknown
    -   Using sample variance as proxy introduces additional error
    -   Conservative: replace z-distribution with t-distribution\
    -   $(t_{n-1,0.01} \times se)$ is the margin of error
    -   Assumptions:
        -   Underlying distribution is Normal
        -   Random Sampling
    -   CI generated from t-distribution are robust wrt normality assumptions violations
-   Sample Size Calculator for Proportions
    -   Margin of error depends on standard error which in turn depends on sample size
    -   Reformulate the CI equation from above
    -   Sample Size : $N = \pi(1-\pi) \times (z^2 / M)$
    -   $\pi$ is the base conversation rate
    -   Z is the Confidence Level
    -   M is the margin of error
-   Sample Size Calculator for Mean
    -   $N = \sigma^2 \times (z^2 / M)$
-   Maximum Likelihood Estimation
    -   Point estimate the maximizes the probability of observed data
    -   Sampling distributions are approximately normal
    -   Use them to estimate variance
-   Bootstrap
    -   Resampling method
    -   Yield standard errors and confidence intervals for measures
    -   No Assumption on underlying distribution

## Significance Test

-   Hypothesis is a statement about the population
-   Significance test uses data to summarize evidence about the hypothesis
-   Five Parts:
    1.  Assumptions
        -   Type of data
        -   Randomization
        -   Population Distribution
        -   Sample Size
    2.  Hypothesis
        -   Null
        -   Alternate
    3.  Test Statistic: How far does the parameter value fall from the hypothesis
    4.  P Value: The probability of observing the given (or more extreme value) of the test statistic, assuming the null hypothesis is true
        -   Smaller the p-value, stronger is the evidence for rejecting null hypothesis
    5.  Conclusion
        -   If P-value is less than 5%, 95% CI doesn't contain the hypothesized value of the parameter
        -   "Reject" or "Fail to Reject" null hypothesis
-   Hypothesis testing for Proportions
    -   $H_0: \pi = \pi_0$
    -   $H_1: \pi \ne \pi_0$
    -   $z = (\hat \pi - \pi_0) / se$
    -   $se = \sqrt{\pi (1-\pi) / N}$
-   Hypothesis testing for Mean
    -   $H_0: \mu = \mu_0$
    -   $H_1: \mu \ne \mu_0$
    -   $t = (\bar X - \mu_0) / se$
    -   $se = \sigma / \sqrt N$
    -   In case of small sample sizes, replace the z-test with binomial distribution
        -   $P(X=x) = {N\choose x} p^x (1-p)^{N-x}$
        -   $\mu = np, \, \sigma=\sqrt{np(1-p)}$
-   One-tail Test measure deviation in a particular direction
    -   Risky in case of skewed distributions
    -   t-test is robust to skewed distributions but one-tailed tests can compound error
    -   Use only when you have a strong directional hypothesis
    -   Provides more power but at the cost of detecting effects in the opposite direction
-   Errors
    -   Type 1: Reject H0, given H0 is true: (1 - Confidence Level)
    -   Type 2: Fail to reject H0, given H0 is false
    -   The smaller P(Type 1 error) is, the larger P(Type 2 error) is.
    -   Probability of Type 2 error increases as statistic moves closer to H0
    -   Power of the test = 1 - P(Type 2 error)
-   Significance testing doesn't rely solely on effect size. Small and impractical differences can be statistically significant with large enough sample sizes

## Comparison of Groups

-   Difference in means between two groups
    -   $\mu_1, \mu_2$ are the average parameter values for the two groups
    -   Test for the difference in $\mu_1 - \mu_2$
    -   Estimate the difference in sample means: $\bar y_1 - \bar y_2$
    -   Assume $\bar y_1 - \bar y_2 \sim N(\mu_1 - \mu_2, se)$
    -   $E(\bar y_1 - \bar y_2) = \mu_1 - \mu_2$
    -   $se = \sqrt{se_1^2 + se_2^2} = \sqrt{s_1^2 / n_1 + s_2^2 / n_2}$
    -   s1 and s2 are standard errors for y1 and y2 respectively
    -   Confidence Intervals
        -   $\bar y_1 - \bar y_2 \pm t (se)$
        -   Check if the confidence interval contains 0 or not
    -   Significance Test
        -   $t= \frac{(\bar y_1 - \bar y_2) - 0}{se}$
        -   degrees of freedom for t is (n1 + n2 -2)
-   Differences in means between two groups (assuming equal variance)
    -   $s = \sqrt{\frac{(n_1 - 1)se_1^2 + (n_2 - 1)se_2^2}{n_1 + n_2 - 2}}$
    -   $se = s \sqrt{{1 \over n_1} + {1 \over n_2}}$
    -   Confidence Interval
        -   $(\bar y_1 - \bar y_2) \pm t (se)$
    -   Significance Test
        -   $t = \frac{(\bar y_1 - \bar y_2)}{se}$\
        -   degrees of freedom for t is (n1 + n2 -2)
-   Difference in proportions between two groups
    -   $\pi_1, \pi_2$ are the average proportion values for the two groups
    -   Test for the difference in $\pi_1 - \pi_2$
    -   $se = \sqrt{se_1^2 + se_2^2} = \sqrt{(\hat\pi_1(1-\hat\pi_1)) / n_1 + (\hat\pi_2(1-\hat\pi_2)) / n_2}$
    -   Confidence Intervals
        -   $\hat \pi_1 - \hat \pi_2 \pm z (se)$
    -   Significance Test
        -   Calculate population average $\hat \pi_1 = \hat \pi_2 = \hat \pi$
        -   $se = \sqrt{\hat\pi(1-\hat\pi)({1 \over n_1} + {1 \over n_2})}$
        -   $z=(\hat \pi_1 - \hat \pi_2) / se$
    -   Fisher's Exact test for smaller samples
-   Differneces in matched pairs
    -   Same subject's response across different times
    -   Controls for other sources of variations
    -   Longitudnal and Crossover studies
    -   Difference of Means == Mean of Differences
    -   Confidence Interval
        -   $\bar y_d \pm t {s_d \over \sqrt n}$
    -   Significance Test
        -   Paired-difference t-test\
        -   $t = {(y_d - 0) \over se}; \; se = s_d / \sqrt n$
    -   Effect Size
        -   $(\bar y_1 - \bar y_2) / s$

| Option | Yes | No  |
|--------|-----|-----|
| Yes    | N11 | N12 |
| No     | N21 | N22 |

-   Comparing Dependent Proportions (McNemar Test)
    -   A 2x2 contingency table (above)
    -   One subject gets multiple treatments
        -   Say disease and side effect (Cancer and Smoking)
    -   $z = \frac{n_{12} - n_{21}}{\sqrt{n_{12} + n_{21}}}$
    -   Confidence Interval
        -   $\hat \pi_1 = (n_{11} + n_{12})/ n$
        -   $\hat \pi_2 = (n_{11} + n_{21}) / n$
        -   $se = {1 \over n}\sqrt{(n_{21} + n_{12}) - (n_{21} + n_{12})^2 / n}$
-   Non-parametric Tests
    -   Wilcoxin Test
        -   Combine Samples n1 + n2
        -   Rank each observation
        -   Compare the mean of the ranks for each group
    -   Mann-Whitney Test
        -   Form pairs of observations from two samples
        -   Count the number of samples in which sample 1 is higher than sample 2

## Association between Categorical Variables

-   Variables are statistically independent if population conditional distributions match the category conditional distribution
-   Chi-Square Test
    -   Calculate Expected Frequencies
    -   (row total \* column total) / total observations
    -   $f_e (xy) = (n_{.y} * n_{x.}) / N$
    -   Compare to observed frequency $f_o$
    -   $\chi^2 = \sum\frac{(f_e - f_o)^2}{f_e}$
    -   degrees of freedom: (r-1)x(c-1)
    -   Value of chi-sq doesn't tell the strength of association
-   Residual Analysis
    -   The difference of a given cell significant or not
    -   $z = (f_e - f_o) / \sqrt{f_e (1 - row\%)(1 - col\%)}$
-   Odds Ratio
    -   Probability of success / Probability of failure
    -   Cross product ratio
    -   From 2x2 Contingecy Tables:
        -   $\theta = (n_{11} \times n_{22}) / (n_{12} \times n_{21})$
    -   $\theta = 1 \implies$ equal probability
    -   $\theta > 1 \implies$ row 1 has higher chance
    -   $\theta < 1 \implies$ row 2 has higher chance
-   Ordinal Variables
    -   Concordance ( C )
        -   Observation higher on one variable is higher on another as well
    -   Discordant ( D )
        -   Otherwise
    -   Calculate Gamma
        -   $\gamma = (C-D) / (C+D)$

## P-value interpretation and common misconceptions:

-   P-value is NOT the probability that the null hypothesis is true
-   P-value is NOT the probability that the results occurred by chance
-   P-value is the probability of obtaining a test statistic at least as extreme as the one observed, given that the null hypothesis is true
-   Small p-values indicate evidence against the null hypothesis, not evidence for the alternative hypothesis 