# Random Forests

## Overview

-   Decision trees have low bias and high variance
-   Bagging or bootstrap aggregation aims to reduce the variance of these classifiers
    -   Average many noisy unbiased models
-   Variance of mean of B random variables (say prediction from trees)
    -   Individual Variance: $\sigma^2$
    -   Pairwise Correlation: $\rho$
    -   $\rho \sigma^2 + {(1 - \rho) \over B} \sigma^2$
    -   Increase in B, cannot cause the forest to overfit.
-   For gains, reduce the correlation between trees
    -   Random feature subset selection
    -   Bootstrap Sampling
-   OOB Error
    -   Errors on observations not selected in Bootstrap Sampling
    -   Identical to CV error

## Variable Importance

-   At each split:
    -   Calculate the improvement in the criterion
    -   Attribute it to the splitting variable
    -   Accumulate over all the trees and splits
-   Using OOB Sample, Permutation accuracy
    -   Pass down OBB samples for a tree, calculate accuracy
    -   Shuffle the variable j
    -   Re-calculate accuracy
    -   Average the difference over all the trees

## Proximity Plots

-   NxN Proximity matrix
-   Count of pairs of OOB observations that share the terminal nodes
-   Multi-dimensional scaling (optional) 