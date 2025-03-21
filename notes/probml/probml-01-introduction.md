# Introduction

-   Definition

    -   A computer program is said to learn from experience E with respect to some class of tasks T, and performance measure P,
    -   If its performance at tasks in T, as measured by P, improves with experience E.

- Supervised Learning
  -   Task is to learn a mapping function from inputs to outputs
  -   Inputs are vectors called features, covariates, predictors
  -   Output is called as label, target, response


-   Classification
    -   Output is a set of unordered, mutually exclusive labels
    -   Goal is to design a model with minimum misclassification rate
    -   $L(\theta) = {1 \over N} \sum I\{y \ne f(x, \theta)\}$
    -   Empirical Risk is the average loss on training set
    -   Model fitting minimizes empirical risk (ERM)
    -   Overall goal is generalization


-   Uncertainty
    -   Model cannot map inputs to outputs with 100% certainty
    -   Model Uncertainty due to lack of knowledge between input and output
    -   Data Uncertainty due to stochasticity in labels
    -   Good model assigns high probability to true output for each input
    -   Intuition for minimizing NLL (negative log-likelihood)
    -   $NLL(\theta) = {1 \over N} \sum p(y | f(x, \theta))$
    -   Optimal parameters give the MLE estimate


-   Regression
    -   Output is a real valued quantity
    -   Model fitting often involves minimizing the quadratic loss or MSE
    -   $L(\theta) = {1 \over N} \sum (y - f(x, \theta))^2$
    -   Data uncertainty. For example: if the output distribution is Gaussian
    -   $p(y | x, \theta) = \mathcal N(y | f(x, \theta), \ \sigma^2)$
    -   $NLL(\theta) \propto MSE(\theta)$
    -   Linear Regression is an affine transformation between inputs and outputs
    -   Polynomial Regression improves the fit by considering higher order interactions
    -   FNNs do feature extraction by nesting the functions


-   Overfitting and Generalization
    -   A model that perfectly fits training data but is too complex suffers from overfitting
    -   Population risk the theoretical expected loss on the true data generating process
    -   Generalization gap is the difference between empirical risk and population risk
    -   High generalization gap is a sign of overfitting
    -   Population risk is hard to estimate. Approximate using test risk. Expected error on unseen data points.
    -   Test error has U-shaped curve wrt model's degree of freedom


-   No Free Lunch Theorem: No single best model that works optimally for all kinds of problems

- Unsupervised Learning
  -   Learn an unconditional model of the data $p(x)$ rather than $p(y | x)$
  -   In clustering, the goal is to partition the input space into regions with homogenous points.
  -   Dimensionality reduction projects input data from high dimension to lower dimension subspace.
  -   Self-Supervised Learning involves creating proxy supervised tasks from unlabled data
  -   Evaluation is done by measuring the probability assigned by the model to unseen data
  -   This treats the problem as one of density estimation
  -   Unsupervised learning also aims to increase sample efficiency in downstream supervised learning tasks

- Reinforcement Learning
  -   A system or an agent learns how to interact with its environment
  -   Goal is learn a policy that specifies optimal action given a state
  -   Unlike, Supervised Learning, the reward signal is occasional and delayed.
  -   Learning with critic vs learning with teacher.

- Data Preprocessing
  -   Text Data
      -   Bag of Words transforms a document to a term frequency matrix
      -   Frequent words have undue influence (pareto distribution)
      -   Log of counts assuage some of the problems
      -   Inverse Document Frequency: $\text{IDF}_i = \log {N \over 1 + \text{DF}_i}$
      -   N is the total number of documents and DF is the documents with term i
      -   $\text{TF-IDF} = \log(1 + TF) \times IDF$
      -   Word embeddings map sparse vector representation of word to lower dimension dense vector
      -   UNK token can help capture OOV words
      -   Sub-word units or word pieces created using byte-pair encoding perform better than UNK token and help in reducing the vocabulary.

  -   Missing Data
      -   MCAR: missing completely at random
      -   MAR: missing at random
      -   NMAR: not missing at random
      -   Handling of the missing values depends to the type 