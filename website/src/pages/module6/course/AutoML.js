import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const AutoML = () => {
  return (
    <Container fluid>
      <h1 className="my-4">AutoML for Tabular Data</h1>

      <section>
        <p>The main goals of AutoML are to:</p>
        <ul>
          <li>Make machine learning accessible to non-experts</li>
          <li>Save time and resources for expert data scientists</li>
          <li>
            Facilitate the exploration of a wider range of models and parameters
          </li>
        </ul>
      </section>

      <section>
        <h2 id="libraries">Popular AutoML Libraries</h2>

        <h3>Auto-sklearn</h3>
        <p>
          Auto-sklearn is an automated machine learning toolkit and a drop-in
          replacement for a scikit-learn estimator.
        </p>
        <CodeBlock
          language="python"
          code={`
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import sklearn.metrics as metrics

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and fit the Auto-sklearn classifier
automl = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30,
                               ensemble_size=1, ensemble_nbest=1)
automl.fit(X_train, y_train)

# Make predictions
y_pred = automl.predict(X_test)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Best model:", automl.show_models())
          `}
        />

        <h3>TPOT</h3>
        <p>
          TPOT is a Python Automated Machine Learning tool that optimizes
          machine learning pipelines using genetic programming.
        </p>
        <CodeBlock
          language="python"
          code={`
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load data
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and fit TPOT classifier
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

# Evaluate the model
print("Accuracy:", tpot.score(X_test, y_test))

# Export the best pipeline
tpot.export('tpot_digits_pipeline.py')
          `}
        />

        <h3>H2O AutoML</h3>
        <p>
          H2O AutoML is an automated machine learning platform that supports
          supervised machine learning tasks like classification and regression.
        </p>
        <CodeBlock
          language="python"
          code={`
import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_iris

# Initialize H2O
h2o.init()

# Load data
iris = load_iris()
iris_df = h2o.H2OFrame(iris.data)
iris_df['species'] = iris.target

# Split the data
train, test = iris_df.split_frame(ratios=[0.8], seed=42)

# Identify predictors and response
x = iris.feature_names
y = 'species'

# Run AutoML
aml = H2OAutoML(max_runtime_secs=120, seed=42)
aml.train(x=x, y=y, training_frame=train)

# Get the best model
best_model = aml.leader

# Evaluate the model
performance = best_model.model_performance(test)
print(performance)

# Shutdown H2O
h2o.shutdown()
          `}
        />
      </section>

      <section>
        <h2 id="implementation">Implementing AutoML Pipelines</h2>
        <p>
          When implementing AutoML pipelines, it's important to consider the
          following steps:
        </p>
        <ol>
          <li>Data preparation and preprocessing</li>
          <li>Feature selection and engineering</li>
          <li>Model selection and hyperparameter tuning</li>
          <li>Ensemble creation</li>
          <li>Model evaluation and interpretation</li>
        </ol>
        <p>
          Here's an example of a custom AutoML pipeline using scikit-learn
          components:
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)),
    ('classifier', RandomForestClassifier())
])

# Define the parameter grid
param_grid = {
    'feature_selection__k': [5, 10, 15],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, None]
}

# Create the grid search object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate on the test set
print("Test set score:", grid_search.score(X_test, y_test))
          `}
        />
      </section>

      <section>
        <h2 id="pros-cons">Pros and Cons of AutoML</h2>
        <h3>Pros</h3>
        <ul>
          <li>Saves time and resources in the model development process</li>
          <li>
            Can discover non-intuitive ML pipelines that humans might overlook
          </li>
          <li>Makes machine learning more accessible to non-experts</li>
          <li>Can produce highly accurate models</li>
        </ul>
        <h3>Cons</h3>
        <ul>
          <li>May require significant computational resources</li>
          <li>
            Can be a "black box", making it harder to understand and explain the
            models
          </li>
          <li>
            May not always produce the best model for specific domain knowledge
          </li>
          <li>Risk of overfitting if not properly validated</li>
        </ul>
      </section>
    </Container>
  );
};

export default AutoML;
