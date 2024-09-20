import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const TreeBasedModels = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Decision Trees</h1>

      <section>
        <h2 id="decision-trees">Decision Trees</h2>
        <h3>Theory and Decision Boundaries</h3>
        <p>
          Decision trees are non-parametric learning methods used for
          classification and regression.
        </p>
        <p>The tree structure consists of:</p>
        <ul>
          <li>Root Node: The topmost node in the tree</li>
          <li>Internal Nodes: Nodes that test an attribute and branch</li>
          <li>
            Leaf Nodes: Terminal nodes that provide the final decision or
            prediction
          </li>
        </ul>

        <h3>Implementation with scikit-learn</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Classification Accuracy: {accuracy_score(y_test, y_pred)}")

# Regression
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"Regression MSE: {mean_squared_error(y_test, y_pred)}")
          `}
        />

        <h3>Key Hyperparameters</h3>
        <ul>
          <li>
            <code>max_depth</code>: Maximum depth of the tree
          </li>
          <li>
            <code>min_samples_split</code>: Minimum number of samples required
            to split an internal node
          </li>
          <li>
            <code>min_samples_leaf</code>: Minimum number of samples required to
            be at a leaf node
          </li>
          <li>
            <code>max_features</code>: Number of features to consider when
            looking for the best split
          </li>
          <li>
            <code>criterion</code>: Function to measure the quality of a split
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default TreeBasedModels;
