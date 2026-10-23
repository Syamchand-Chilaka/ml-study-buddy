# Machine Learning Fundamentals — AIML-500 Study Guide

## What is Machine Learning?

Machine learning (ML) is a way of teaching computers to learn patterns from
examples instead of following hand-written rules. In traditional programming,
a person writes the rules and the computer applies them to data. In machine
learning, the computer is given data and the desired answers, and it figures
out the rules on its own.

Think of it like teaching a child to recognize dogs. You don't hand them a
rulebook ("four legs, fur, barks"). You show them many pictures of dogs and
non-dogs, and over time they learn the pattern themselves. That is exactly
what an ML model does with training data.

The three main ingredients of any ML project are: data (the examples),
a model (the pattern-learner), and an objective (what "good" means, measured
by a loss function).

## AI vs Machine Learning

Artificial intelligence (AI) is the broad field of making computers do
things that normally require human intelligence — understanding language,
recognizing images, making decisions. Machine learning is a subset of AI:
the approach where the computer learns those abilities from data instead of
being explicitly programmed. Deep learning is a further subset of ML that
uses many-layered neural networks.

Think of nested circles: AI is the biggest circle, machine learning sits
inside it, and deep learning sits inside machine learning. A rule-based
chess program is AI but not ML (humans wrote its rules); a spam filter that
learns from examples is both AI and ML.

## Types of Machine Learning

### Supervised Learning

In supervised learning, every training example comes with a label — the
correct answer. The model learns to map inputs to outputs. It is like
studying with an answer key: you practice on questions where you can check
yourself. Most business applications of ML are supervised: spam filtering,
price prediction, medical image screening.

### Unsupervised Learning

In unsupervised learning, there are no labels. The model looks for structure
on its own, such as grouping similar customers together (clustering) or
compressing data into fewer dimensions. It is like sorting a box of mixed
LEGO bricks into piles without being told what the piles should be.

### Reinforcement Learning

In reinforcement learning, an agent learns by trial and error, receiving
rewards or penalties for its actions. It is like training a puppy with
treats: good actions get rewarded, and over many attempts the agent learns
a strategy (called a policy) that maximizes reward. This powers game-playing
AI and robotics.

## Classification vs Regression

Both classification and regression are supervised learning tasks, but they
differ in what they predict.

Classification predicts a category (a discrete label). "Is this email spam
or not spam?" "Is this tumor benign or malignant?" "Which of these 10 digits
is in the image?" The output is one of a fixed set of classes. Think of it
as a sorting hat: every input gets placed into exactly one bucket.

Regression predicts a number (a continuous value). "What will this house
sell for?" "What temperature will it be tomorrow?" "How many units will we
sell next month?" The output can be any value on a number line. Think of it
as a dial rather than a bucket.

A quick test to tell them apart: if the answer is a word or category, it is
classification; if the answer is a quantity you could plot on a ruler, it is
regression.

## Training, Validation, and Test Data

A model must be evaluated on data it has never seen. The standard practice
is to split the dataset into three parts:

- Training set: the examples the model learns from.
- Validation set: used during development to tune settings (hyperparameters).
- Test set: held back until the very end, to give an honest estimate of
  real-world performance.

Evaluating on training data is like grading students on the exact homework
problems they already practiced — everyone looks like a genius, and the
grade means nothing about new problems.

## Overfitting and Underfitting

Overfitting happens when a model memorizes the training data — including its
noise and quirks — instead of learning the general pattern. The model scores
brilliantly on training data but fails on new data.

Analogy: a student who memorizes the answers to last year's exam word-for-word.
They ace a retake of that exact exam, but the moment the questions change
slightly, they fall apart. They memorized answers; they never learned the
subject.

Underfitting is the opposite: the model is too simple to capture the pattern
at all. It does poorly on training data AND new data. That is the student who
only skimmed the chapter titles — no amount of question-changing helps,
because they never learned enough in the first place.

Ways to reduce overfitting include: getting more training data, simplifying
the model, regularization (penalizing complexity), early stopping, and
cross-validation. The sweet spot between underfitting and overfitting is
called good generalization — this is the whole goal of machine learning.

## The Bias-Variance Tradeoff

Bias is error from a model being too simple (systematically wrong).
Variance is error from a model being too sensitive to its particular
training data (wildly different results if you re-train on new samples).
Simple models tend to have high bias and low variance; complex models tend
to have low bias and high variance. Good modeling means balancing the two.

## Evaluation Metrics

### Accuracy — and why it can mislead

Accuracy is the fraction of predictions that are correct. It is intuitive,
but it can be deeply misleading on imbalanced datasets, where one class
vastly outnumbers the other.

The classic example is fraud detection. Suppose only 1% of credit card
transactions are fraudulent. A lazy model that predicts "not fraud" for
every single transaction achieves 99% accuracy — while catching exactly
zero fraud. The metric looks superb; the model is useless. The same trap
appears in rare-disease screening and defect detection.

With imbalanced classes, accuracy mostly reflects the majority class, so you
must look at metrics that focus on the rare class: precision, recall, and F1.

### Precision, Recall, and F1 Score

Precision answers: of everything the model flagged as positive, how much
really was positive? (How trustworthy are its alarms?)

Recall answers: of all the real positives out there, how many did the model
catch? (How much slips past it?)

There is a tradeoff: a smoke detector that goes off at the faintest whiff
has high recall (misses nothing) but low precision (many false alarms).
F1 score is the harmonic mean of precision and recall — a single number
that is only high when both are reasonably high.

### The Confusion Matrix

A confusion matrix is a 2x2 table showing true positives, false positives,
true negatives, and false negatives. It is the most honest picture of a
classifier's behavior, because it shows exactly which kinds of mistakes
the model makes, not just how many.

## Features and Feature Engineering

Features are the input variables a model uses to make predictions — the
columns of your dataset. Feature engineering is the craft of creating,
transforming, and selecting features so the model can learn more easily.
For example, turning a raw timestamp into "day of week" and "is holiday"
often helps a sales model far more than the raw number ever could.
Practitioners often say: better features beat fancier algorithms.

## Common Algorithms at a Glance

- Linear regression: fits a straight line; the workhorse of regression.
- Logistic regression: despite the name, a classification algorithm; outputs
  probabilities.
- Decision trees: a flowchart of yes/no questions; easy to explain.
- Random forests: many decision trees voting together; robust and popular.
- k-nearest neighbors: classifies a point by looking at its closest examples.
- Neural networks: layers of simple units that together learn very complex
  patterns; the foundation of deep learning.

## Neural Networks

A neural network is a model made of layers of simple units called neurons.
Each neuron takes numbers in, multiplies them by weights, adds them up, and
passes the result on. One neuron is trivial; millions of them stacked in
layers can learn astonishingly complex patterns.

Analogy: think of a huge committee of very simple voters. Each voter only
looks at a tiny piece of the evidence and casts a weighted vote; layer after
layer of voters combine those votes into a final decision. Training the
network means repeatedly adjusting the weights — turning up the voters that
helped get answers right and turning down the ones that misled — using an
algorithm called backpropagation.

Networks with many layers are called deep neural networks, which is where
the term deep learning comes from. They power image recognition, speech
systems, and large language models.

## Key Terms Recap

- Model: the learned pattern-mapper from inputs to outputs.
- Label: the correct answer attached to a training example.
- Loss function: the score the model tries to minimize during training.
- Hyperparameter: a setting chosen by the developer (like tree depth),
  not learned from data.
- Generalization: performing well on data the model has never seen —
  the ultimate goal.
