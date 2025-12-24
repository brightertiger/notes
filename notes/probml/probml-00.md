# Probabilistic Machine Learning Notes

These notes are based on "Probabilistic Machine Learning" by Kevin Murphy — a comprehensive modern treatment of machine learning from a probabilistic perspective. This guide makes these concepts accessible to undergraduates while maintaining the depth needed for practical understanding.

## What is Probabilistic Machine Learning?

Probabilistic ML treats machine learning as a problem of **inference under uncertainty**. Instead of just making predictions, we quantify how confident we are in those predictions. This is crucial for real-world applications where decisions have consequences.

**Key idea**: Everything is uncertain — our data is noisy, our models are approximations, and we never have enough data. Probability theory gives us a principled framework to reason about this uncertainty.

## Why the Probabilistic Perspective?

1. **Quantified Uncertainty**: Know when to trust your model's predictions
2. **Principled Learning**: Derive optimal learning algorithms from first principles
3. **Regularization**: Prevent overfitting through priors and Bayesian inference
4. **Model Comparison**: Compare different models in a principled way
5. **Decision Making**: Make optimal decisions under uncertainty

## Topics Covered

| Topic | What You'll Learn |
|-------|-------------------|
| **Probability** | Foundation of uncertainty quantification |
| **Statistics** | Inference, estimation, and hypothesis testing |
| **Decision Theory** | Making optimal choices under uncertainty |
| **Information Theory** | Measuring information and uncertainty |
| **Optimization** | Finding the best model parameters |
| **Discriminant Analysis** | Generative vs. discriminative models |
| **Linear & Logistic Regression** | Foundational supervised learning |
| **Neural Networks** | Deep learning architectures (FFN, CNN, RNN) |
| **Trees & Ensembles** | Decision trees, random forests, boosting |
| **Exemplar Methods** | KNN, metric learning |
| **Self-Supervised Learning** | Learning from unlabeled data |
| **Recommendation Systems** | Collaborative filtering and matrix factorization |

## Prerequisites

To get the most from these notes:
- **Calculus**: Derivatives, gradients, chain rule
- **Linear Algebra**: Matrices, eigenvalues, matrix decompositions
- **Basic Probability**: Random variables, expectations, common distributions
- **Programming**: Python with NumPy, familiarity with ML libraries helpful

## How to Use These Notes

1. **Start with foundations**: Probability and statistics chapters build the foundation
2. **Understand the "why"**: Focus on intuition before equations
3. **Connect concepts**: Many ideas recur across chapters (e.g., MLE, regularization)
4. **Practice**: Implement algorithms to solidify understanding

Let's dive in!
