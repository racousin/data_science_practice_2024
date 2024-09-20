import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import DataSplitDiagram from "components/module6/DataSplitDiagram";

const ModelSelection = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Selection Techniques</h1>

      <section>
        <h2>Train-Test Split</h2>
        <DataSplitDiagram />
        <p>
          The train-test split method is fundamental for evaluating machine
          learning model performance. It involves:
        </p>

        <ol>
          <li>
            Randomly partitioning the dataset into two subsets: training and
            test sets
          </li>
          <li>Training the model on the training set</li>
          <li>Evaluating the model's performance on the test set</li>
        </ol>

        <p>
          This approach ensures unbiased performance estimation by using
          separate data for training and evaluation.
        </p>
      </section>
      <section>
        <h2 id="cross-validation">Cross-validation Strategies</h2>
        <p>
          Cross-validation is a resampling procedure used to evaluate machine
          learning models on a limited data sample. It helps to assess how the
          model will generalize to an independent dataset.
        </p>

        <h3>K-Fold Cross-validation</h3>
        <Row className="justify-content-center">
          <Col xs={12} md={10} lg={8}>
            <div className="text-center">
              <Image
                src="/assets/module6/kfold-cross-validation.svg"
                alt="kfold-cross-validation"
                fluid
              />
              <p>kfold-cross-validation</p>
            </div>
          </Col>
        </Row>
        <p>
          K-Fold CV involves splitting the dataset into k subsets, then for each
          subset:
          <ul>
            <li>Take the subset as a hold out or test data set</li>
            <li>Take the remaining subsets as a training data set</li>
            <li>
              Fit a model on the training set and evaluate it on the test set
            </li>
            <li>Retain the evaluation score and discard the model</li>
          </ul>
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
svm = SVC(kernel='rbf', random_state=42)

scores = cross_val_score(svm, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())
          `}
        />

        <h3>Stratified K-Fold</h3>
        <p>
          Stratified K-Fold is a variation of K-Fold that returns stratified
          folds: each set contains approximately the same percentage of samples
          of each target class as the complete set.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(svm, X, y, cv=skf)
print("Stratified K-Fold scores:", scores)
print("Mean Stratified K-Fold score:", scores.mean())
          `}
        />

        <h3>Leave-One-Out Cross-validation</h3>
        <p>
          Leave-One-Out CV uses a single observation from the original sample as
          the validation data, and the remaining observations as the training
          data. This is repeated such that each observation in the sample is
          used once as the validation data.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(svm, X, y, cv=loo)
print("Leave-One-Out CV mean score:", scores.mean())
          `}
        />

        <h3>Time Series Cross-validation</h3>
        <p>
          For time series data, it's important to respect the temporal order of
          observations. Time Series Split provides train/test indices to split
          time series data samples that are observed at fixed time intervals.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(svm, X, y, cv=tscv)
print("Time Series CV scores:", scores)
print("Mean Time Series CV score:", scores.mean())
          `}
        />
      </section>

      <section>
        <h2 id="nested-cv">Nested Cross-validation</h2>
        <p>
          Nested cross-validation is used to train a model in which
          hyperparameters also need to be optimized. It consists of an inner
          loop CV for hyperparameter tuning and an outer loop CV for evaluating
          the model performance.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.model_selection import GridSearchCV, cross_val_score

# Define the parameter grid
param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

# Set up the nested CV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create the grid search object
clf = GridSearchCV(SVC(random_state=42), param_grid, cv=inner_cv)

# Perform nested CV
nested_scores = cross_val_score(clf, X, y, cv=outer_cv)

print("Nested CV scores:", nested_scores)
print("Mean nested CV score:", nested_scores.mean())
          `}
        />
      </section>
    </Container>
  );
};

export default ModelSelection;
