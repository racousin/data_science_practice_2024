import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const EnsembleTechniques = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Ensemble Techniques</h1>

      <section>
        <h2 id="bagging">Bagging (Bootstrap Aggregating)</h2>
        <p>
          Bagging is an ensemble technique that creates multiple subsets of the
          original dataset using random sampling with replacement (bootstrap
          sampling), trains a model on each subset, and then aggregates the
          predictions.
        </p>
        <h3>Key Characteristics:</h3>
        <ul>
          <li>Reduces variance and helps to avoid overfitting</li>
          <li>
            Works well with high-variance, low-bias models (e.g., decision
            trees)
          </li>
          <li>Trains models in parallel</li>
        </ul>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = DecisionTreeClassifier(random_state=42)
bagging_model = BaggingClassifier(base_estimator=base_model, n_estimators=10, random_state=42)

bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
print(f"Bagging Classifier Accuracy: {accuracy_score(y_test, y_pred)}")
          `}
        />
      </section>

      <section>
        <h2 id="stacking">Stacking</h2>
        <p>
          Stacking (Stacked Generalization) is an ensemble technique that
          combines multiple models via a meta-learner. It trains several base
          models and then uses another model to learn how to best combine the
          predictions of the base models.
        </p>
        <h3>Theory and Implementation</h3>
        <p>The process of stacking typically involves:</p>
        <ol>
          <li>Split the training data into two parts</li>
          <li>Train several base models on the first part</li>
          <li>Make predictions on the second part</li>
          <li>
            Use the predictions from step 3 as inputs to train a higher-level
            learner
          </li>
        </ol>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

# Define meta-learner
meta_learner = LogisticRegression(random_state=42)

# Create stacking ensemble
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

# Fit the stacking ensemble
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred_stack = stacking_model.predict(X_test)
print(f"Stacking Classifier Accuracy: {accuracy_score(y_test, y_pred_stack)}")
          `}
        />

        <h3>Choosing Base Models and Meta-learners</h3>
        <p>When choosing models for stacking:</p>
        <ul>
          <li>
            Base models should be diverse (different algorithms, different
            hyperparameters)
          </li>
          <li>
            Base models should have reasonably good performance individually
          </li>
          <li>
            The meta-learner should be able to learn the strengths and
            weaknesses of the base models
          </li>
          <li>
            Logistic Regression is a common choice for the meta-learner due to
            its simplicity and interpretability
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default EnsembleTechniques;
