import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const RegressionModels = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Regression Models</h1>
      <p>
        In this section, you will learn about various regression models and how
        to implement and evaluate them.
      </p>
      <Row>
        <Col>
          <h2>Linear Regression</h2>
          <p>
            Linear regression is a simple and widely used regression model. It
            assumes a linear relationship between the input features and the
            target variable.
          </p>
          <CodeBlock
            code={`# Example of linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          />
          <h2>Polynomial Regression</h2>
          <p>
            Polynomial regression is a type of regression model that assumes a
            polynomial relationship between the input features and the target
            variable.
          </p>
          <CodeBlock
            code={`# Example of polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)`}
          />
          <h2>Regularization Techniques</h2>
          <p>
            Regularization techniques are used to prevent overfitting in
            regression models. Examples include Ridge regression and Lasso
            regression.
          </p>
          <CodeBlock
            code={`# Example of Ridge regression
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default RegressionModels;
