import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const LinearModels = () => {
  <section>
    <p>
      Linear models are fundamental in machine learning and serve as a starting
      point for many analyses. They offer good interpretability and can be
      extended with regularization techniques to prevent overfitting. While they
      may not capture complex non-linear relationships, they provide a solid
      baseline and are often used in combination with other models in ensemble
      methods.
    </p>
  </section>;
  return (
    <Container fluid>
      <h1 className="my-4">Linear Models</h1>

      <section>
        <h2 id="linear-regression">Linear Regression</h2>
        <p>
          Linear regression models the relationship between a dependent variable
          y and one or more independent variables X, assuming a linear
          relationship.
        </p>
        <BlockMath math="y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon" />
        <p>Key assumptions of linear regression:</p>
        <ul>
          <li>Linearity: The relationship between X and y is linear</li>
          <li>Independence: Observations are independent of each other</li>
          <li>Homoscedasticity: Constant variance of residuals</li>
          <li>Normality: Residuals are normally distributed</li>
        </ul>

        <h3>Implementation with scikit-learn</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_[0]}")
print(f"Intercept: {model.intercept_[0]}")
print(f"Mean squared error: {mse}")
print(f"R-squared score: {r2}")
          `}
        />

        <h3>Key Hyperparameters</h3>
        <p>
          Linear Regression has few hyperparameters. The main ones in
          scikit-learn's implementation are:
        </p>
        <ul>
          <li>
            <code>fit_intercept</code>: Whether to calculate the intercept
            (default is True)
          </li>
          <li>
            <code>normalize</code>: Whether to normalize the input features
            (default is False)
          </li>
        </ul>
      </section>

      <section>
        <h2 id="logistic-regression">Logistic Regression</h2>
        <h3>Binary and Multiclass Classification</h3>
        <p>
          Logistic regression is used for classification problems. It models the
          probability of an instance belonging to a particular class.
        </p>
        <BlockMath math="P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}" />

        <h3>Implementation with scikit-learn</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
          `}
        />

        <h3>Key Hyperparameters</h3>
        <ul>
          <li>
            <code>C</code>: Inverse of regularization strength (smaller values
            specify stronger regularization)
          </li>
          <li>
            <code>penalty</code>: Type of regularization ('l1', 'l2',
            'elasticnet', or 'none')
          </li>
          <li>
            <code>solver</code>: Algorithm to use in the optimization problem
          </li>
          <li>
            <code>max_iter</code>: Maximum number of iterations for solvers to
            converge
          </li>
        </ul>
      </section>

      <section>
        <h2 id="regularized-models">Regularized Linear Models</h2>
        <h3>Lasso, Ridge, and Elastic Net</h3>
        <p>
          Regularization helps prevent overfitting by adding a penalty term to
          the loss function.
        </p>
        <ul>
          <li>
            Ridge Regression (L2):{" "}
            <BlockMath math="\text{Loss} = \text{MSE} + \lambda \sum_{j=1}^p \beta_j^2" />
          </li>
          <li>
            Lasso Regression (L1):{" "}
            <BlockMath math="\text{Loss} = \text{MSE} + \lambda \sum_{j=1}^p |\beta_j|" />
          </li>
          <li>Elastic Net: Combination of L1 and L2</li>
        </ul>

        <h3>Trade-off between Bias and Variance</h3>
        <p>
          Regularization introduces a bias-variance trade-off. It increases the
          bias of the model but reduces its variance, which can lead to better
          generalization on unseen data.
        </p>

        <h3>Implementation and Hyperparameters</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 20)
y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Ridge MSE: {mse_ridge}")

# Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"Lasso MSE: {mse_lasso}")

# Elastic Net
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
print(f"Elastic Net MSE: {mse_elastic}")
          `}
        />

        <h3>Key Hyperparameters for Regularized Models</h3>
        <ul>
          <li>
            <code>alpha</code>: Regularization strength (higher values increase
            regularization)
          </li>
          <li>
            <code>l1_ratio</code>: The Elastic Net mixing parameter (0 &lt;=
            l1_ratio &lt;= 1)
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default LinearModels;
