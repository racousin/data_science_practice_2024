import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const EnsembleModels = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Ensemble Models</h1>

      <section>
        <h2 id="boosting">Boosting</h2>
        <p>
          Boosting is a family of algorithms that convert weak learners into
          strong learners. It builds models sequentially, where each new model
          tries to correct the errors of the previous ones.
        </p>
        <h3>AdaBoost (Adaptive Boosting)</h3>
        <p>
          AdaBoost adjusts the weights of misclassified instances after each
          iteration, focusing subsequent models on the hard-to-classify
          instances.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import AdaBoostClassifier

adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)
y_pred_ada = adaboost_model.predict(X_test)
print(f"AdaBoost Classifier Accuracy: {accuracy_score(y_test, y_pred_ada)}")
          `}
        />

        <h3>Gradient Boosting</h3>
        <p>
          Gradient Boosting builds models sequentially, with each new model
          trying to minimize the loss function of the entire ensemble using
          gradient descent.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print(f"Gradient Boosting Classifier Accuracy: {accuracy_score(y_test, y_pred_gb)}")
          `}
        />
      </section>

      <section>
        <h2 id="random-forests">Random Forests</h2>
        <h3>Ensemble Learning Concept</h3>
        <p>
          Random Forests are an ensemble learning method that operate by
          constructing multiple decision trees during training and outputting
          the class that is the mode of the classes (classification) or mean
          prediction (regression) of the individual trees.
        </p>
        <p>Key concepts:</p>
        <ul>
          <li>
            Bagging (Bootstrap Aggregating): Training each tree on a random
            subset of the data
          </li>
          <li>
            Feature Randomness: Considering only a random subset of features for
            splitting at each node
          </li>
        </ul>

        <h3>Implementation and Key Hyperparameters</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print(f"Random Forest Classification Accuracy: {accuracy_score(y_test, y_pred)}")

# Regression
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)
print(f"Random Forest Regression MSE: {mean_squared_error(y_test, y_pred)}")
          `}
        />
      </section>

      <section>
        <h2 id="gradient-boosting">Gradient Boosting Machines</h2>
        <p>
          Gradient Boosting is a machine learning technique that produces a
          prediction model in the form of an ensemble of weak prediction models,
          typically decision trees.
        </p>

        <h3>XGBoost, LightGBM, and CatBoost</h3>
        <p>These are popular implementations of gradient boosting:</p>
        <ul>
          <li>
            <strong>XGBoost</strong>: Optimized distributed gradient boosting
            library
          </li>
          <li>
            <strong>LightGBM</strong>: Light Gradient Boosting Machine, uses
            histogram-based algorithms
          </li>
          <li>
            <strong>CatBoost</strong>: Implements ordered boosting and an
            innovative algorithm for processing categorical features
          </li>
        </ul>

        <h3>Implementation and Key Hyperparameters</h3>
        <CodeBlock
          language="python"
          code={`
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

# LightGBM
lgbm = LGBMClassifier(n_estimators=100, random_state=42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgbm)}")

# CatBoost
cb = CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
cb.fit(X_train, y_train)
y_pred_cb = cb.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred_cb)}")
          `}
        />

        <ul>
          <li>
            CatBoost can handle categorical variables natively using its ordered
            boosting algorithm
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default EnsembleModels;
