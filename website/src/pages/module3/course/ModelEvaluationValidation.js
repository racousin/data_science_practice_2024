import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";

// import DataSplittingDiagram from "components/module3/DataSplittingDiagram";
// import ModelEvaluationFlowchart from "components/module3/ModelEvaluationFlowchart";
// import ErrorComparisonChart from "components/module3/ErrorComparisonChart";
// import DatasetPartitioningVenn from "components/module3/DatasetPartitioningVenn";
// import GeneralizationErrorConcept from "components/module3/GeneralizationErrorConcept";

const ModelEvaluationValidation = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Evaluation</h1>
      <p>
        Model evaluation is the process of assessing how well a machine learning
        model performs. It's a crucial step in the machine learning pipeline for
        several reasons:
      </p>
      <ul>
        <li>
          It helps us understand if our model is learning meaningful patterns
          from the data.
        </li>
        <li>
          It allows us to compare different models and choose the best one for
          our problem.
        </li>
        <li>
          It provides insights into how the model might perform in real-world
          scenarios.
        </li>
      </ul>

      <h2>Model Fitting and Inference</h2>

      <p>In supervised learning, we typically have:</p>
      <ul>
        <li>
          <InlineMath math="X \in \mathbb{R}^{n \times p}" />: Input features (n
          samples, p features)
        </li>
        <li>
          <InlineMath math="y \in \mathbb{R}^n" />: Target variable
        </li>
        <li>
          <InlineMath math="f: \mathbb{R}^p \rightarrow \mathbb{R}" />: The true
          function we're trying to approximate
        </li>
        <li>
          <InlineMath math="\hat{f}: \mathbb{R}^p \rightarrow \mathbb{R}" />:
          Our model's approximation of f
        </li>
      </ul>

      <p>
        The goal is to find <InlineMath math="\hat{f}" /> that minimizes some
        loss function <InlineMath math="L(y, \hat{f}(X))" />.
      </p>

      <h3>Model Fitting</h3>
      <p>
        Model fitting, or training, is the process of finding the best
        parameters for our model <InlineMath math="\hat{f}" /> using our
        training data.
      </p>
      <CodeBlock
        language="python"
        code={`
import numpy as np
from sklearn.linear_model import LinearRegression

# Assume X and y are our feature matrix and target vector
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression()
model.fit(X, y)  # This is where the model learns from the data
        `}
      />

      <h3>Inference</h3>
      <p>
        Once the model is trained, we can use it to make predictions on new
        data:
      </p>
      <CodeBlock
        language="python"
        code={`
X_new = np.array([[3, 5], [4, 4]])
y_pred = model.predict(X_new)
print("Predictions:", y_pred)
        `}
      />

      <h2>2. Model Evaluation and Data Splitting</h2>
      <p>
        To properly evaluate a model's performance and ensure unbiased estimates
        of its generalization ability, we need to carefully partition our data.
        This partitioning is crucial for understanding how well our model learns
        and generalizes.
      </p>

      <h3>The Need for Data Splitting</h3>
      <p>
        When evaluating a model, we aim to estimate its performance on unseen
        data. If we use the same data for training and evaluation, we risk
        overly optimistic estimates due to the model memorizing the training
        data rather than learning generalizable patterns. To mitigate this, we
        split our data into distinct sets:
      </p>
      <p>
        Let D be our complete dataset. We partition D into three disjoint
        subsets:
      </p>
      <BlockMath math="D = D_{train} \cup D_{val} \cup D_{test}" />
      <BlockMath math="D_{train} \cap D_{val} = D_{train} \cap D_{test} = D_{val} \cap D_{test} = \emptyset" />

      <h3>Training Set</h3>
      <p>
        The training set is used to fit the model, allowing it to learn the
        underlying patterns in the data. Typically, it comprises 60-80% of the
        total dataset.
      </p>

      <h3>Validation Set</h3>
      <p>The validation set serves two primary purposes:</p>
      <ol>
        <li>
          Tuning hyperparameters: We use this set to optimize model parameters
          that are not learned during training.
        </li>
        <li>
          Model selection: When comparing different models or architectures, the
          validation set helps us choose the best performing one.
        </li>
      </ol>
      <p>The validation set usually consists of 10-20% of the total dataset.</p>

      <h3>Test Set</h3>
      <p>
        The test set provides an unbiased estimate of the model's performance on
        new, unseen data. It's crucial to keep this set completely separate from
        the training process to avoid data leakage. The test set typically
        comprises 10-20% of the total dataset.
      </p>

      <h3>Implementation</h3>
      <p>In practice, we often use a two-step splitting process:</p>
      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import train_test_split

# Step 1: Separate test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Split remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
  `}
      />

      <h3>Evaluation Process</h3>
      <p>
        With this splitting strategy, our model evaluation process follows these
        steps:
      </p>
      <ol>
        <li>Train the model on D_train</li>
        <li>Evaluate and tune the model using D_val</li>
        <li>Once the model is finalized, assess its performance on D_test</li>
      </ol>

      <p>
        This approach allows us to obtain unbiased estimates of model
        performance and make informed decisions about model selection and
        hyperparameter tuning.
      </p>

      <h2>3. Understanding Model Performance</h2>
      <p>
        Now that we understand how to split our data, let's discuss how we
        measure a model's performance.
      </p>

      <h3>Training Error</h3>
      <p>
        The training error is the error that our model makes on the training
        data. A low training error means our model fits the training data well,
        but it doesn't necessarily mean it will perform well on new data.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_squared_error

y_train_pred = model.predict(X_train)
train_error = mean_squared_error(y_train, y_train_pred)
print(f"Training Error: {train_error}")
  `}
      />

      <h3>Validation Error</h3>
      <p>
        The validation error is the error on our validation set. We use this to
        tune our model and make decisions about which model or hyperparameters
        to use.
      </p>
      <CodeBlock
        language="python"
        code={`
y_val_pred = model.predict(X_val)
val_error = mean_squared_error(y_val, y_val_pred)
print(f"Validation Error: {val_error}")
  `}
      />

      <h3>Test Error</h3>
      <p>
        The test error is the error on our test set. This gives us an estimate
        of how well our model will perform on new, unseen data. It's crucial
        that we only use the test set once we've finalized our model to get an
        unbiased estimate of its performance.
      </p>
      <CodeBlock
        language="python"
        code={`
y_test_pred = model.predict(X_test)
test_error = mean_squared_error(y_test, y_test_pred)
print(f"Test Error: {test_error}")
  `}
      />

      <h3>Generalization Error</h3>
      <p>
        The generalization error is the expected error on new, unseen data. We
        estimate this using our test error, but the true generalization error
        can only be known if we had access to all possible data.
      </p>
      <p>
        In practice, we use the test error as our best estimate of the
        generalization error:
      </p>
      <CodeBlock
        language="python"
        code={`
estimated_generalization_error = test_error
print(f"Estimated Generalization Error: {estimated_generalization_error}")
  `}
      />
      <p>
        Note: The true generalization error would require evaluating the model
        on the entire population of possible data points, which is typically not
        feasible.
      </p>
      <h2>4. Overfitting and Underfitting</h2>
      <p>
        Understanding the concepts of overfitting and underfitting is crucial
        for building models that generalize well.
      </p>

      <h3>Overfitting</h3>
      <p>
        Overfitting occurs when a model learns the training data too well,
        including its noise and fluctuations. An overfit model has:
      </p>
      <ul>
        <li>Low training error, high validation/test error</li>
        <li>Complex model that captures noise in the data</li>
        <li>Poor generalization to new data</li>
      </ul>

      <h3>Underfitting</h3>
      <p>
        Underfitting occurs when a model is too simple to capture the underlying
        patterns in the data. An underfit model has:
      </p>
      <ul>
        <li>High training error, high validation/test error</li>
        <li>Overly simple model that fails to capture important patterns</li>
        <li>Poor performance on both training and new data</li>
      </ul>

      <h2>5. Bias-Variance Tradeoff</h2>
      <p>
        The expected generalization error of a model can be decomposed into
        three components:
      </p>
      <BlockMath math="E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2" />
      <p>Where:</p>
      <ul>
        <li>
          <InlineMath math="y" /> is the true value
        </li>
        <li>
          <InlineMath math="\hat{f}(x)" /> is the model's prediction
        </li>
        <li>
          <InlineMath math="\sigma^2" /> is the irreducible error
        </li>
      </ul>

      <p>
        The goal is to find the sweet spot where the combined error from bias
        and variance is minimized.
      </p>
      <h3>Bias</h3>
      <p>
        Bias is the error introduced by approximating a real-world problem,
        which may be complex, by a simplified model. High bias can lead to
        underfitting.
      </p>

      <h3>Variance</h3>
      <p>
        Variance is the error introduced by the model's sensitivity to small
        fluctuations in the training set. High variance can lead to overfitting.
      </p>

      <h2>6. Cross-Validation Techniques</h2>
      <p>
        Cross-validation is a resampling procedure used to evaluate machine
        learning models on a limited data sample. It provides a more robust
        estimate of model performance than a single train-test split.
      </p>

      <h3>K-Fold Cross-Validation</h3>
      <p>
        In K-Fold CV, the data is divided into k subsets. The model is trained
        on k-1 subsets and validated on the remaining subset. This process is
        repeated k times, with each subset serving as the validation set once.
      </p>

      <CodeBlock
        language="python"
        code={`
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())
        `}
      />

      <h3>Stratified K-Fold Cross-Validation</h3>
      <p>
        Stratified K-Fold CV is a variation of K-Fold that ensures that the
        proportion of samples for each class is roughly the same in each fold as
        in the whole dataset. This is particularly useful for imbalanced
        datasets.
      </p>

      <h2>7. Time Series Cross-Validation</h2>
      <p>
        Time series data presents unique challenges for cross-validation due to
        its temporal nature. Traditional random splitting can lead to data
        leakage and overly optimistic performance estimates.
      </p>

      <h3>Time Series Split</h3>
      <p>
        Time Series Split is a variation of K-Fold CV that respects the temporal
        order of the data. It creates training-validation splits by
        incrementally adding samples to the training set and using the next
        chunk of data as the validation set.
      </p>

      <h3>Rolling Window Validation</h3>
      <p>
        Rolling Window Validation uses a fixed-size window that moves through
        the time series data. This approach is particularly useful when you want
        to maintain a consistent training set size and capture recent trends.
      </p>
    </Container>
  );
};

export default ModelEvaluationValidation;
