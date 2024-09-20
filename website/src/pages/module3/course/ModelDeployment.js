import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ModelDeployment = () => {
  return (
    <Container fluid>
      <h1 className="my-4">From Model Evaluation to Deployment</h1>

      <section>
        <h2 id="versioning-tracking">
          Model Versioning and Experiment Tracking
        </h2>
        <p>
          As machine learning projects evolve, it's crucial to keep track of
          different model versions, hyperparameters, and experiment results.
          This practice enables reproducibility and facilitates collaboration
          among team members.
        </p>

        <h3>Tools for Experiment Tracking</h3>
        <ul>
          <li>
            <strong>MLflow:</strong> An open source platform for the complete
            machine learning lifecycle
          </li>
          <li>
            <strong>Weights & Biases:</strong> A tool for experiment tracking,
            dataset versioning, and model management
          </li>
          <li>
            <strong>DVC (Data Version Control):</strong> Version control system
            for machine learning projects
          </li>
        </ul>

        <h3>Example: Using MLflow for Experiment Tracking</h3>
        <CodeBlock
          language="python"
          code={`
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Enable automatic logging of pandas, sklearn, and numpy operations
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Your machine learning code here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(rf, "random_forest_model")
          `}
        />
      </section>

      <section>
        <h2 id="ab-testing">A/B Testing in Production</h2>
        <p>
          A/B testing, also known as split testing, is a method of comparing two
          versions of a model or system to determine which one performs better
          in a real-world environment.
        </p>

        <h3>Key Considerations for A/B Testing</h3>
        <ul>
          <li>Define clear metrics for success</li>
          <li>Ensure statistical significance</li>
          <li>Control for external factors</li>
          <li>Monitor for unexpected side effects</li>
        </ul>

        <h3>Example: Implementing A/B Testing</h3>
        <CodeBlock
          language="python"
          code={`
import numpy as np
from scipy import stats

def ab_test(control_conversions, control_size, 
            treatment_conversions, treatment_size,
            confidence_level=0.95):
    
    # Calculate conversion rates
    control_rate = control_conversions / control_size
    treatment_rate = treatment_conversions / treatment_size
    
    # Calculate pooled standard error
    se = np.sqrt(control_rate * (1 - control_rate) / control_size + 
                 treatment_rate * (1 - treatment_rate) / treatment_size)
    
    # Calculate z-score
    z_score = (treatment_rate - control_rate) / se
    
    # Calculate p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Check if result is statistically significant
    is_significant = p_value < (1 - confidence_level)
    
    return {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'difference': treatment_rate - control_rate,
        'z_score': z_score,
        'p_value': p_value,
        'is_significant': is_significant
    }

# Example usage
result = ab_test(control_conversions=100, control_size=1000,
                 treatment_conversions=120, treatment_size=1000)
print(result)
          `}
        />
      </section>

      <section>
        <h2 id="performance-monitoring">Monitoring Model Performance</h2>
        <p>
          Once a model is deployed, it's crucial to continuously monitor its
          performance to ensure it maintains its predictive power over time.
        </p>

        <h3>Key Metrics to Monitor</h3>
        <ul>
          <li>Model accuracy or error rates</li>
          <li>Input data distribution shifts</li>
          <li>Prediction latency</li>
          <li>Resource utilization (CPU, memory, etc.)</li>
        </ul>

        <h3>Example: Setting Up Basic Model Monitoring</h3>
        <CodeBlock
          language="python"
          code={`
import numpy as np
from sklearn.metrics import accuracy_score
import time

class ModelMonitor:
    def __init__(self, model, expected_accuracy=0.9, max_latency=0.1):
        self.model = model
        self.expected_accuracy = expected_accuracy
        self.max_latency = max_latency
        self.predictions = []
        self.actual = []
        self.latencies = []

    def predict(self, X, y_true):
        start_time = time.time()
        y_pred = self.model.predict(X)
        end_time = time.time()

        self.predictions.extend(y_pred)
        self.actual.extend(y_true)
        self.latencies.append(end_time - start_time)

        return y_pred

    def check_performance(self):
        accuracy = accuracy_score(self.actual, self.predictions)
        avg_latency = np.mean(self.latencies)

        print(f"Current Accuracy: {accuracy:.4f}")
        print(f"Average Latency: {avg_latency:.4f} seconds")

        if accuracy < self.expected_accuracy:
            print("WARNING: Model accuracy below expected threshold")
        if avg_latency > self.max_latency:
            print("WARNING: Average latency above maximum threshold")

# Usage
monitor = ModelMonitor(model, expected_accuracy=0.9, max_latency=0.1)
y_pred = monitor.predict(X_test, y_test)
monitor.check_performance()
          `}
        />
      </section>

      <section>
        <h2 id="update-strategies">
          Strategies for Model Updates and Retraining
        </h2>
        <p>
          As new data becomes available and the underlying patterns in the data
          change, it's important to have strategies in place for updating and
          retraining models.
        </p>

        <h3>Retraining Strategies</h3>
        <ul>
          <li>
            <strong>Periodic Retraining:</strong> Retrain the model at fixed
            intervals
          </li>
          <li>
            <strong>Performance-based Retraining:</strong> Retrain when
            performance drops below a threshold
          </li>
          <li>
            <strong>Online Learning:</strong> Continuously update the model as
            new data arrives
          </li>
          <li>
            <strong>Ensemble Updates:</strong> Add new models to an ensemble
            over time
          </li>
        </ul>

        <h3>Example: Implementing a Simple Retraining Strategy</h3>
        <CodeBlock
          language="python"
          code={`
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

class RetraniningClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier, retraining_threshold=0.9):
        self.base_classifier = base_classifier
        self.retraining_threshold = retraining_threshold
        self.n_retrained = 0

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.base_classifier.predict(X)

    def score(self, X, y):
        accuracy = accuracy_score(y, self.predict(X))
        if accuracy < self.retraining_threshold:
            print(f"Accuracy {accuracy:.4f} below threshold. Retraining...")
            self.fit(self.X_, self.y_)
            self.n_retrained += 1
        return accuracy

# Usage
from sklearn.tree import DecisionTreeClassifier
clf = RetraniningClassifier(DecisionTreeClassifier(), retraining_threshold=0.9)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(f"Final Accuracy: {accuracy:.4f}")
print(f"Number of retraining iterations: {clf.n_retrained}")
          `}
        />
      </section>
    </Container>
  );
};

export default ModelDeployment;
