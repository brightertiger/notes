# SSL

-   Data Augmentation
    -   Artificially modified versions of input vectors that may appear in real world data
    -   Improves accuracy, makes model robust
    -   Empirical risk minimization to vicinal risk minimization
    -   Minimizing risk in the vicinity of input data point

-   Transfer Learning 
    - Some data poor tasks may have structural similarity to other data rich tasks 
    - Transferring information from one dataset to another via shared parameters of a model 
    - Pretrain the model on a large source dataset 
    - Fine tune the model on a small target dataset 
    - Chop-off the head of the pretrained model and add a new one 
    - The parameters may be frozen during fine-tuning 
    - In case the parameters aren't frozen, use small learning rates. 
  
-   Adapters 
    - Modify the model structure to customize feature extraction 
    - For example: Add MLPs after transformer blocks and initialize them for identity mappings 
    - Much less parameters to be learned during fine-tuning
  
-   Pre-training
    -   Can be supervised or unsupervised.
    -   Supervised
        -   Imagenet is supervised pretraining.
        -   For unrelated domains, less helpful.
        -   More like speedup trick with a good initialization.
    -   Unsupervised
        -   Use unlabeled dataset
        -   Minimize reconstruction error
    -   Self-supervised
        -   Labels are created from unlabeled dataset algorithmically
        -   Cloze Task
            -   Fill in the blanks
        -   Proxy Tasks
            -   Create representations
            -   Siamese Neural Networks
            -   Capture relationship between inputs
        -   Contrastive Tasks
            -   Use data augmentation
            -   Ensure that similar inputs have closer representations
    -   SimCLR
        -   Simple Contrastive Learning for Visual Representations
        -   Pretraining
            -   Take an unlabeled image X
            -   Apply data augmentation A and A' and get two views X and X'
            -   Apply encoder to both views
            -   Apply projection head to both encodings
            -   Contrastive Loss function with mini-batch to identify the positive pairs
        -   Fine-tuning
            -   Finetune the encoder and a task specific head
            -   The original projection head is discarded
        -   NLP Encoders
            -   Contrastively trained language models
            -   SimCSE: Simple Contrastive Learning for Sentence Embeddings
                -   Dropout acts as data augmentation
                -   Cosine Similarity between representations
    -   Non-Contrastive Learning
        -   BYOL
            -   Bootstrap your own latent (BYOL)
            -   Two copies of encoder
            -   Online network (student): trained with gradient descent
            -   Target network (teacher): EMA of online network weights
            -   MSE Loss between online prediction and target representation

-   Semi-Supervised Learning
    -   Learn from labeled + unlabeled data in tandem
    -   Self-Training
        -   Train the model on labeled data
        -   Run inference on unlabeled data to get predictions (pseudo-labels, machine generated)
        -   Combine machine generated and human generated labels
        -   Self-training is similar to EM algorithm where pseudo-labels are the E-step
    -   Noise Student Training
        -   Adds noise to student model to improve generalization
        -   Add noise via dropout, stochastic depth, data augmentation
    -   Consistency Regularization
        -   Model's prediction shouldn't change much for small changes to the input
        -   Can be implemented by passing augmented versions of same image as loss
    -   Label Propagation
        -   Graph: Nodes are i/p datapoints and edges denote similarity
        -   Use graph clustering to group related nodes
        -   Class labels are assigned to unlabeled data, based on the cluster distribution
        -   Model Labels: Labels of the data points
        -   Propagate the labels in such a way that there is minimal label disagreement between node and it's neighbours
        -   Label guesses for unlabeled data that can be used for supervised learning
        -   Details:
            -   M labeled points, N unlabeled points
            -   T: (M+N) x (M+N) transition matrix of normalized edge weights
            -   Y: Label matrix for class distribution of (M+N) x C dimension
            -   Use transition matrix to propagate labels Y = TY until convergence
        -   Success depends on calculating similarity between data points
    -   Consistency Regularization
        -   Small perturbation to input data point should not change the model predicitons
-   Generative Models
    -   Natural way of using unlabeled data by learning a model of data generative process.
    -   Variational Autoencoders
        -   Models joint distribution of data (x) and latent variables (z)
        -   First sample: $z \sim p(z)$ and then sample $x\sim p(x|z)$
        -   Encoder: Approximate the posterior
        -   Decoder: Approximate the likelihood
        -   Maximize evidence lower bound of the data (ELBO) (derived from Jensen's inequality)
        -   Use VAEs to learn representations for downstream tasks
    -   Generative Adversarial Networks
        -   Generator: Maps latent distribution to data space
        -   Discriminator: Distinguish between outputs of generator and true distribution
        -   Modify discriminator to predict class labels and fake rather than just fake
-   Active Learning
    -   Identify true predictive mapping by querying as few data points as possible
    -   Query Synthesis: Model asks output for any input
    -   Pool Based: Model selects the data point from a pool of unlabeled data points
    -   Maximum Entropy Sampling
        -   Uncertainty in predicted label
        -   Fails when examples are ambiguous of mislabeled
    -   Bayesian Active Learning by Disagreement (BALD)
        -   Select examples where model makes predictions that are highly diverse
-   Few-Shot Learning
    -   Learn to predict from very few labeled example
    -   One-Shot Learning: Learn to predict from single example
    -   Zero-Shot Learning: Learn to predict without labeled examples
    -   Model has to generalize for unseen labels during traning time
    -   Works by learning a distance metric
-   Weak Supervision
    -   Exact label not available for data points
    -   Distribution of labels for each case
    -   Soft labels / label smoothing 

# Semi-Supervised and Self-Supervised Learning

- Semi-Supervised Learning (SSL): Leveraging both labeled and unlabeled data
  - Motivation: Labels are expensive, unlabeled data is abundant
  - Assumption: Underlying data distribution contains useful structure
  
- Data Augmentation
  - Creates artificial training examples through transformations
  - Preserves semantic content while changing surface features
  - Common augmentations:
    - Image domain: rotations, flips, color jitter, cropping
    - Text domain: synonym replacement, back-translation
    - Audio domain: pitch shifting, time stretching
  - Theoretical framework: Vicinical risk minimization
    - Minimize risk in local neighborhoods around training examples
    - Improves robustness and generalization
  
- Transfer Learning
  - Leverages knowledge from data-rich domains to improve performance in data-poor domains
  - Process:
    1. Pretrain model on large source dataset (e.g., ImageNet, Common Crawl)
    2. Adapt model to target task with smaller dataset
    3. Options for adaptation:
       - Feature extraction: Freeze pretrained layers, train only new head
       - Fine-tuning: Update all or subset of pretrained parameters
  - Parameter-efficient fine-tuning:
    - Adapters: Small bottleneck layers added between frozen transformer blocks
    - LoRA: Low-rank adaptation of weight matrices
    - Prompt tuning: Learn soft prompts while keeping model parameters frozen

- Self-Supervised Learning
  - Creates supervisory signals from unlabeled data
  - Pretext tasks:
    - Reconstruction tasks: Autoencoders, masked language modeling
    - Context prediction: Predict arrangement of shuffled patches
    - Contrastive tasks: Learn similar representations for related inputs
  
  - Contrastive Learning
    - Learn representations by comparing similar and dissimilar examples
    - SimCLR framework:
      1. Generate two views of each image via augmentation
      2. Encode both views with shared encoder
      3. Apply projection head to map encodings to space for contrastive loss
      4. Contrastive loss: Maximize similarity between positive pairs (same image)
         and minimize similarity between negative pairs (different images)
      5. For downstream tasks, discard projection head and fine-tune encoder
    
    - Key challenges:
      - Hard negative mining: Finding informative negative examples
      - Batch size dependence: Performance scales with number of negatives
      - Feature collapse: Trivial solutions that ignore semantic content

  - Non-Contrastive Methods
    - BYOL (Bootstrap Your Own Latent):
      - Teacher-student architecture with no negative examples
      - Student network predicts teacher network outputs
      - Teacher parameters updated via exponential moving average of student
      - Avoids collapse through asymmetric architecture and predictor networks
    
    - Masked Autoencoders:
      - Inspired by BERT's success in NLP
      - Mask significant portions of input (e.g., 75% of image patches)
      - Train encoder-decoder to reconstruct original input
      - For downstream tasks, use only encoder

- Practical Considerations
  - Pretraining often provides:
    - Better initialization for optimization
    - More generalizable features
    - Sample efficiency: Fewer labeled examples needed
  - Domain gap between pretraining and target task affects transfer effectiveness
  - Large pretrained models may contain useful knowledge but require careful adaptation 