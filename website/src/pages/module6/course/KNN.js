import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const KNN = () => {
  return (
    <Container fluid>
      <h1 className="my-4">K-Nearest Neighbors (KNN)</h1>

      <section>
        <h2 id="theory">Theory and Distance Metrics</h2>
        <p>
          K-Nearest Neighbors (KNN) is a simple and non-parametric algorithm
          used for both classification and regression tasks. The core idea is to
          predict the target variable of a new data point based on the values of
          its K nearest neighbors in the feature space.
        </p>
        <h3>Algorithm Steps:</h3>
        <ol>
          <li>Choose the number K of neighbors</li>
          <li>
            Calculate the distance between the query instance and all training
            samples
          </li>
          <li>Sort the distances and determine the K nearest neighbors</li>
          <li>
            For classification: Use majority vote of the neighbors' labels
          </li>
          <li>For regression: Use the mean of the neighbors' values</li>
        </ol>

        <h3>Distance Metrics</h3>
        <p>
          The choice of distance metric is crucial in KNN. Common distance
          metrics include:
        </p>
        <ul>
          <li>
            Euclidean Distance:{" "}
            <BlockMath math="d(p,q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}" />
          </li>
          <li>
            Manhattan Distance:{" "}
            <BlockMath math="d(p,q) = \sum_{i=1}^n |p_i - q_i|" />
          </li>
          <li>
            Minkowski Distance:{" "}
            <BlockMath math="d(p,q) = (\sum_{i=1}^n |p_i - q_i|^r)^{\frac{1}{r}}" />
          </li>
        </ul>
      </section>

      <section>
        <h2 id="implementation">Implementation and Key Hyperparameters</h2>
        <CodeBlock
          language="python"
          code={`
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Classification example
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)
y_pred_clf = knn_clf.predict(X_test_scaled)
print(f"KNN Classification Accuracy: {accuracy_score(y_test, y_pred_clf)}")

# Regression example
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Regressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)
y_pred_reg = knn_reg.predict(X_test_scaled)
print(f"KNN Regression MSE: {mean_squared_error(y_test, y_pred_reg)}")
          `}
        />

        <h3>Key Hyperparameters</h3>
        <ul>
          <li>
            <code>n_neighbors</code>: Number of neighbors to use (K)
          </li>
          <li>
            <code>weights</code>: Weight function used in prediction ('uniform',
            'distance', or a callable function)
          </li>
          <li>
            <code>metric</code>: Distance metric to use ('euclidean',
            'manhattan', 'minkowski', etc.)
          </li>
          <li>
            <code>p</code>: Power parameter for the Minkowski metric
          </li>
          <li>
            <code>algorithm</code>: Algorithm used to compute nearest neighbors
            ('auto', 'ball_tree', 'kd_tree', or 'brute')
          </li>
        </ul>
      </section>

      {/* <section>
        <h2 id="pros-cons">Pros and Cons of Lazy Learning</h2>
        <h3>Pros</h3>
        <ul>
          <li>Simple to understand and implement</li>
          <li>No assumptions about the underlying data distribution</li>
          <li>Can be used for both classification and regression</li>
          <li>Can capture complex decision boundaries</li>
          <li>No training phase (lazy learning)</li>
        </ul>
        <h3>Cons</h3>
        <ul>
          <li>
            Computationally expensive for large datasets (need to compute
            distances to all training samples)
          </li>
          <li>Requires feature scaling</li>
          <li>
            Sensitive to irrelevant features and the curse of dimensionality
          </li>
          <li>Needs to store all training data</li>
          <li>Choosing the optimal value of K can be challenging</li>
        </ul>
      </section> */}
    </Container>
  );
};

export default KNN;
