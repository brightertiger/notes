# Model Assessment and Selection

## Bias-Variance Tradeoff

-   Generalization
    -   Prediction error over an independent test sample
-   $\text{ERR}_T = E[L(Y, \hat f(x)) | T]$
    -   T refers to the training set used to build the model
    -   L is the loss function used to evaluate the model performance
        -   Regression: Squared Loss, Absolute Loss
        -   Classification: 0-1 Loss, Deviance (-2 × log-likelihood)
-   As the model becomes more complex, it adapts to complex underlying structure of the training data
    -   Decrease in bias but increase in variance
    -   If the underlying training data changes, the complex fitted model will change to a large extent
-   Intermediate model complexity gives minimum expected test error
    -   U-shaped test error curve: underfitting → optimal → overfitting
-   Training error is not a good estimate of test error
    -   Consistently decreases with increasing model complexity
    -   Poor generalization
    -   Training error is an optimistic estimate of test error
-   Model Selection
    -   Estimating performance of different models to choose the best one
    -   Select the model with best estimated generalization performance
-   Model Assessment
    -   Having selected the model, estimating the generalization error on new, unseen data
    -   Provides unbiased estimate of model performance
-   Divide the dataset
    -   Training: Fit the models
    -   Validation: Estimate model prediction error for model selection
    -   Test: Generalization error of the final chosen model
    -   Typically 50%/25%/25% or 60%/20%/20% split for train/validation/test

## Bias-Variance Decomposition

-   $Y = f(X) + \epsilon$
    -   $E(\epsilon) = 0, Var(\epsilon) = \sigma^2_{\epsilon}$
-   $\text{ERR}(x_0) = E[(Y - \hat f(x_0))^2 | x_0]$
    -   $\text{ERR}(x_0) = E[(f(x_0) + \epsilon - \hat f(x_0))^2]$
    -   $\text{ERR}(x_0) = \sigma^2_{\epsilon} + [E(\hat f(x_0)) - f(x_0)]^2 + E[(\hat f(x_0) - E(\hat f(x_0)))^2]$
    -   MSE = Irreducible Error + Bias Squared + Variance
-   Bias: Difference between average of estimate and true mean
    -   Bias is high for simple models (underfitting)
    -   Simpler models make strong assumptions about underlying structure
-   Variance: Squared Deviation of model around its mean
    -   Variance is high for complex models (overfitting)
    -   Complex models are sensitive to specific training samples
-   More Complex Model
    -   Lower Bias
    -   Higher Variance
    -   Tradeoff depends on signal-to-noise ratio and training sample size
-   For linear Model
    -   $\text{Variance} \propto p$
        -   Complexity of the model is related to the number of parameters
    -   $\text{Bias}^2 = \text{Model Bias}^2 + \text{Estimation Bias}^2$
        -   Model Bias: Best fitting linear model and True function
        -   Estimation Bias: Estimated Model and Best fitting linear model
    -   For OLS: Estimation Bias is 0, BLUE
    -   For Ridge: Estimation Bias is positive
        -   Trade-off with reduction in variance
        -   Can lead to overall lower MSE despite increased bias

## Optimism of Training Error

-   $\text{ERR} = E(\text{ERR}_T)$
-   Training error is less than test error
    -   Same data is being used to train and evaluate the model
    -   Model is optimized to fit training data
-   Optimistic estimate of generalization error
    -   $\text{ERR}_{in}$:
        -   Error between sample and population regression function estimates on training data
    -   $\bar{\text{err}}$
        -   Average sample regression error over training data
-   Optimism in training error estimate
    -   $\text{op} = \text{ERR}_{in} - \bar{\text{err}}$
    -   Related to $\text{cov}(y, \hat y)$
    -   How strongly a label value affects its own prediction
    -   For linear smoothers: $\text{op} = \frac{2}{N}\sum_{i=1}^{N}\text{cov}(y_i, \hat{y}_i)$
-   Optimism increases with number of inputs (model complexity)
-   Optimism decreases with number of training samples
    -   Larger samples dilute the influence of individual points

## In-sample Prediction Error

-   $\text{ERR}_{in} = \bar{\text{err}} + \text{op}$
-   Cp Statistic
    -   $C_p = \bar{\text{err}} + 2{p \over N} \sigma^2_{\epsilon}$
    -   p is the effective number of parameters
    -   Mallow's Cp provides unbiased estimate of prediction error
-   AIC (Akaike Information Criterion)
    -   $\text{AIC} = {-2 \over N} LL + 2{p \over N}$
    -   p is the effective number of parameters
    -   For model selection, choose the one with lowest AIC
    -   Information-theoretic interpretation: minimizing KL divergence
    -   Tends to select more complex models as N grows
-   Effective Number of Parameters
    -   Linear Regression: $\hat y = ((X'X)^{-1}X')y$
    -   Ridge Regression: $\hat y = (((X'X)^{-1} + \lambda I)X')y$
    -   Generalized Form: $\hat y = S y$
    -   p is the trace of the S Matrix
    -   For Ridge: $\text{df}(\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$
-   BIC (Bayesian Information Criterion)
    -   $\text{BIC} = {-2 \over N} LL + \log N \times p$
    -   Penalizes complex models more heavily compared to AIC
    -   Consistent: will select true model as N → ∞ (if true model is in the set)
    -   Tends to select simpler models than AIC
    -   Bayesian Approach
        -   $P(M |D) \propto P(D |M) P(M)$
    -   Laplace Approximation
        -   $\log P(D |M) = \log P(D |M, \theta) - p \log N$
        -   $\log P(D |M, \theta)$ is the MLE objective function
    -   Compare two models
        -   $P(M1 |D) / P(M2 |D) = P(M1) / P(M2) + P(D | M1) / P(D | M2)$
        -   The first term is constant (non-informative priors)
        -   The second term is the Bayes Factor

## VC Dimension

-   AIC, C-p statistic need the information on model complexity
    -   Effective number of parameters
-   Difficult to estimate for non-linear models
-   VC Dimension is Generalized Model Complexity of a class of functions
    -   How "wiggly" can the member of this class be?
    -   Vapnik-Chervonenkis dimension: maximum number of points that can be shattered
-   Shattering
    -   Points that can be perfectly separated by a class of functions, no matter how the binary labels are assigned
    -   2^n possible labelings must all be achievable
-   VC Dimension: Largest number of points that can be shattered by members of class of functions
    -   VC(linear functions in d dimensions) = d+1
    -   VC(polynomials of degree d) = d+1
    -   VC(neural networks) depends on architecture
-   3 points in case of linear classifier in a plane
    -   4 points can lead to XOR problem (not linearly separable)
-   Structural Risk Minimization
    -   Upper bound on generalization error: $\text{error} \leq \text{training error} + \sqrt{\frac{h(\log(2N/h) + 1) - \log(\eta/4)}{N}}$
    -   Where h is VC dimension and η is confidence parameter

## Cross Validation

-   Estimation for $\text{ERR}_T$
-   Data scarce situation
-   Divide data into K equal parts
    -   Indexing function: $\kappa : \{1,2,....N\} \rightarrow \{1, 2 ... K\}$
-   Fit model on K-1 parts and predict on Kth part
-   Cross Validation Error
    -   $CV(f) = {1 \over N}\sum L(y_i, \hat y_i^{f_{-\kappa}})$
    -   Average of test error across all folds
-   Types of CV:
    -   K=N: Leave-one-out CV (LOOCV)
       - Low bias but high variance
       - Computationally expensive for large datasets
    -   K=5 or 10: k-fold CV
       - Balance between bias and variance
       - Computationally efficient
    -   K=2: Repeated random subsampling
       - Can have high variance
-   5-fold, 10-fold cross validation is recommended
    -   Empirical studies show best balance of bias and variance
    -   Higher values of K give more biased estimates

## Bootstrap Methods

-   Estimation for $\text{ERR}$
-   Randomly draw datasets from training data by sampling with replacement
    -   Each dataset has the same size as original training data
    -   Each bootstrap sample contains ~63.2% of unique original observations
-   Fit the model on each of the bootstrap datasets
-   $\text{ERR}_{\text{boot}} = \frac{1}{B}\frac{1}{N} \sum_b \sum_i L(y_i, \hat f^b(x_i))$
-   Bootstrap uses overlapping samples across model fits (unlike cross validation)
    -   $P(i \in B) = 1 - (1 - \frac{1}{N})^N \approx 1 - e^{-1} \approx 0.632$
-   $\text{ERR}_{\text{boot}}$ isn't a good estimator because of leakage
    -   Observations used in both training and testing
-   Use Out-of-bag error instead
    -   Evaluate on samples not included in each bootstrap sample
    -   For each observation i, average predictions from models where i wasn't used
    -   Provides nearly unbiased estimate of test error
-   .632 Bootstrap estimator
    -   $\text{ERR}_{.632} = 0.368 \cdot \text{err}_{\text{train}} + 0.632 \cdot \text{err}_{\text{oob}}$
    -   Correction for bias in both training error and OOB estimates 