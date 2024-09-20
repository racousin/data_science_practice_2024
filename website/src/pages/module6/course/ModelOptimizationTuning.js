import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ModelOptimizationTuning = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Optimization and Tuning</h1>
      <p>
        In this section, you will learn about techniques for improving the
        performance of machine learning models.
      </p>
      <Row>
        <Col>
          <h2>Hyperparameter Tuning using Grid Search and Random Search</h2>
          <p>
            Hyperparameters are parameters that are not learned from the data
            and need to be set manually. Grid search and random search are
            techniques for finding the optimal hyperparameters for a model.
          </p>
          <CodeBlock
            code={`# Example of hyperparameter tuning using grid search
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)`}
          />
          <h2>Feature Importance and Model Simplification</h2>
          <p>
            Feature importance measures the contribution of each feature to the
            model's predictions. By identifying the most important features, it
            is possible to simplify the model and improve its interpretability.
          </p>
          <CodeBlock
            code={`# Example of feature importance using a random forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()`}
          />
          <h2>Ensemble Methods (Boosting, Bagging, Stacking)</h2>
          <p>
            Ensemble methods combine multiple models to improve their
            performance. Examples include boosting, bagging, and stacking.
          </p>
          <CodeBlock
            code={`# Example of boosting using gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ModelOptimizationTuning;
