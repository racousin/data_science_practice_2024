import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";
import { BlockMath, InlineMath } from "react-katex";
import CodeBlock from "components/CodeBlock";

const EvaluationMetrics = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Evaluation Metrics in Machine Learning</h1>

      <p>
        Evaluation metrics are crucial in assessing the performance of machine
        learning models. They provide quantitative measures of how well a
        model's predictions align with the actual outcomes. The choice of metric
        depends on the type of problem being solved and the specific goals of
        the project.
      </p>

      <p>
        Evaluation metrics quantify the difference between the target variable{" "}
        <InlineMath math="y" /> and the model's predictions{" "}
        <InlineMath math="\hat{y}" />, providing a measure of how well our model
        is performing.
      </p>
      <h2>Choosing the Right Metric</h2>
      <p>The choice of evaluation metric depends on various factors:</p>
      <ul>
        <li>
          Nature of the problem (regression, classification, ranking, etc.)
        </li>
        <li>Business objectives and cost of errors</li>
        <li>Class balance in classification problems</li>
        <li>Sensitivity to outliers</li>
      </ul>
      <h2>Types of Prediction Problems and Their Metrics</h2>

      <h3>1. Regression Metrics</h3>
      <p>
        Regression problems involve predicting continuous numerical values.
        These could range from predicting house prices to estimating a person's
        age from a photo. The choice of metric can significantly impact how we
        interpret model performance.
      </p>

      <h4>Mean Squared Error (MSE)</h4>
      <p>
        MSE measures the average squared difference between predicted and actual
        values. It gives higher weight to larger errors, making it particularly
        sensitive to outliers.
      </p>
      <BlockMath math="MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> MSE is in squared units, which can be
        hard to interpret directly. It's also sensitive to the scale of the
        target variable - an MSE of 100 is large for age prediction but small
        for house price prediction.
      </p>

      <h4>Root Mean Squared Error (RMSE)</h4>
      <p>
        RMSE is the square root of MSE, providing a metric in the same unit as
        the target variable. This makes it more interpretable than MSE.
      </p>
      <BlockMath math="RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  `}
      />
      <p>
        <strong>Characteristic:</strong> Like MSE, RMSE is sensitive to
        outliers. It's also not always the best choice if you care equally about
        all errors, regardless of their magnitude.
      </p>

      <h4>Mean Absolute Error (MAE)</h4>
      <p>
        MAE measures the average absolute difference between predicted and
        actual values. It treats all errors equally, regardless of their
        magnitude.
      </p>
      <BlockMath math="MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> MAE is more robust to outliers than
        MSE/RMSE. However, it doesn't penalize large errors as heavily, which
        might be important in some applications.
      </p>

      <h4>R-squared (Coefficient of Determination)</h4>
      <p>
        R-squared represents the proportion of variance in the dependent
        variable that is predictable from the independent variable(s). It ranges
        from 0 to 1, with 1 indicating perfect prediction.
      </p>
      <BlockMath math="R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> R-squared can be misleading for
        non-linear relationships. It also tends to increase as you add more
        variables to your model, even if those variables aren't truly
        predictive.
      </p>

      <h4>Dealing with Dimensionality</h4>
      <p>
        When working with multi-dimensional regression problems (predicting
        multiple continuous values simultaneously), these metrics can be applied
        in different ways:
      </p>
      <ul>
        <li>
          <strong>Average across dimensions:</strong> Calculate the metric for
          each dimension separately and then take the average. This gives equal
          weight to each dimension.
        </li>
        <li>
          <strong>Treat as single dimension:</strong> Flatten the predictions
          and true values into a single vector before calculating the metric.
          This can be misleading if the dimensions have different scales.
        </li>
        <li>
          <strong>Weighted average:</strong> Calculate the metric for each
          dimension and take a weighted average based on the importance of each
          dimension.
        </li>
      </ul>
      <p>
        <strong>Example:</strong> Imagine you're predicting both the width and
        height of an object. You could calculate RMSE separately for width and
        height and then average them. Alternatively, you could treat each
        width-height pair as a single prediction and calculate RMSE on the
        flattened data.
      </p>
      <CodeBlock
        language="python"
        code={`
import numpy as np
from sklearn.metrics import mean_squared_error

# Example multi-dimensional predictions
y_true = np.array([[1, 2], [3, 4], [5, 6]])
y_pred = np.array([[1.1, 2.1], [3.2, 3.9], [4.8, 6.2]])

# Method 1: Average across dimensions
rmse_dim1 = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
rmse_dim2 = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
rmse_avg = (rmse_dim1 + rmse_dim2) / 2

# Method 2: Treat as single dimension
rmse_flat = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

print(f"Average RMSE: {rmse_avg}")
print(f"Flattened RMSE: {rmse_flat}")
  `}
      />
      <p>
        Choosing the right approach depends on your specific problem and how you
        want to interpret the results. Always consider the context and potential
        implications of your choice.
      </p>

      <h3>2. Binary Classification Metrics</h3>
      <p>
        Binary classification involves predicting one of two possible outcomes.
        These could range from predicting whether an email is spam or not, to
        determining if a medical test result is positive or negative. The choice
        of metric can significantly impact how we interpret model performance,
        especially in situations with imbalanced classes.
      </p>

      <h4>Accuracy</h4>
      <p>
        Accuracy is the ratio of correct predictions to total predictions. It's
        a straightforward metric but can be misleading for imbalanced datasets.
      </p>
      <BlockMath math="Accuracy = \frac{TP + TN}{TP + TN + FP + FN}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> Accuracy is intuitive but can be
        misleading for imbalanced datasets. For example, if 95% of emails are
        not spam, a model that always predicts "not spam" would have 95%
        accuracy but would be useless for detecting spam.
      </p>

      <h4>Precision</h4>
      <p>
        Precision is the ratio of true positive predictions to total positive
        predictions. It answers the question: "Of all the instances predicted as
        positive, how many actually are positive?"
      </p>
      <BlockMath math="Precision = \frac{TP}{TP + FP}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> Precision is particularly important
        when the cost of false positives is high. For example, in a spam
        detection system, high precision means fewer legitimate emails are
        incorrectly classified as spam.
      </p>

      <h4>Recall (Sensitivity)</h4>
      <p>
        Recall is the ratio of true positive predictions to total actual
        positives. It answers the question: "Of all the actual positive
        instances, how many were correctly identified?"
      </p>
      <BlockMath math="Recall = \frac{TP}{TP + FN}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> Recall is crucial when the cost of
        false negatives is high. In medical diagnosis, for instance, high recall
        ensures that few actual positive cases are missed, even if it means more
        false positives.
      </p>

      <h4>F1 Score</h4>
      <p>
        F1 Score is the harmonic mean of precision and recall, providing a
        single score that balances both metrics.
      </p>
      <BlockMath math="F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}" />
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> F1 Score is particularly useful when
        you have an uneven class distribution. It helps to find an optimal blend
        of precision and recall. However, it doesn't distinguish between the
        types of errors (false positives vs. false negatives).
      </p>

      <h4>ROC AUC (Receiver Operating Characteristic Area Under Curve)</h4>
      <p>
        ROC AUC represents the model's ability to distinguish between classes.
        The ROC curve is created by plotting the True Positive Rate against the
        False Positive Rate at various threshold settings.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_true, y_pred_proba)
  `}
      />
      <Row className="justify-content-center">
        <Col xs={12} md={10} lg={8}>
          <div className="text-center">
            <Image src="/assets/module3/roc_auc.png" alt="roc_auc" fluid />
            <p>roc_auc</p>
          </div>
        </Col>
      </Row>
      <p>
        <strong>Characteristic:</strong> ROC AUC is scale-invariant and
        classification-threshold-invariant. It measures the quality of the
        model's predictions irrespective of what classification threshold is
        chosen. However, when working with highly imbalanced datasets,
        Precision-Recall AUC might be more informative.
      </p>

      <h4>Dealing with Imbalanced Datasets</h4>
      <p>
        In many real-world scenarios, the classes in a binary classification
        problem are not evenly distributed. This class imbalance can
        significantly impact the interpretation and usefulness of these metrics.
      </p>
      <ul>
        <li>
          <strong>Balanced Accuracy:</strong> The average of recall obtained on
          each class. It's useful for imbalanced datasets.
        </li>
        <li>
          <strong>Precision-Recall Curve:</strong> A plot of the Precision vs
          the Recall at various threshold settings. The area under this curve
          (PR AUC) can be more informative than ROC AUC for imbalanced datasets.
        </li>
        <li>
          <strong>Matthews Correlation Coefficient (MCC):</strong> A metric that
          takes into account all four confusion matrix categories (TP, TN, FP,
          FN), providing a balanced measure even for imbalanced datasets.
        </li>
      </ul>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_true, y_pred)

# Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"Balanced Accuracy: {balanced_acc}")
print(f"Matthews Correlation Coefficient: {mcc}")
print(f"Precision-Recall AUC: {pr_auc}")
  `}
      />
      <p>
        Choosing the right metric(s) depends on your specific problem, the class
        distribution in your dataset, and the relative importance of different
        types of errors in your application. Always consider multiple metrics to
        get a comprehensive view of your model's performance.
      </p>

      <h3>3. Multi-class Classification Metrics</h3>
      <p>
        Multi-class classification involves predicting one of three or more
        possible outcomes. These could range from classifying images into
        multiple categories to predicting the genre of a book. The choice of
        metric becomes more complex as we need to consider performance across
        all classes.
      </p>

      <h4>Macro-averaged F1 Score</h4>
      <p>
        Macro-averaged F1 Score calculates the F1 score for each class
        independently and then takes the unweighted mean. This treats all
        classes as equally important, regardless of their frequency in the
        dataset.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import f1_score

macro_f1 = f1_score(y_true, y_pred, average='macro')
  `}
      />
      <p>
        <strong>Characteristic:</strong> Macro-averaged F1 is useful when you
        want to give equal importance to the performance on all classes, even if
        some classes are rare. However, it can be sensitive to performance on
        infrequent classes.
      </p>

      <h4>Weighted-averaged F1 Score</h4>
      <p>
        Weighted-averaged F1 Score calculates the F1 score for each class and
        then takes the weighted mean based on the number of samples in each
        class. This accounts for class imbalance.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import f1_score

weighted_f1 = f1_score(y_true, y_pred, average='weighted')
  `}
      />
      <p>
        <strong>Characteristic:</strong> Weighted-averaged F1 is appropriate
        when you want to account for label imbalance. It gives more weight to
        classes with more samples, which might be desirable if the class
        distribution in your test set matches the real-world distribution.
      </p>

      <h4>Dealing with Multi-class Imbalance</h4>
      <p>
        Multi-class classification often involves imbalanced datasets, which can
        lead to biased models and misleading evaluation metrics. Some strategies
        to address this include:
      </p>
      <ul>
        <li>
          <strong>Confusion Matrix:</strong> Visualize the performance across
          all classes.
        </li>
        <li>
          <strong>Per-class Metrics:</strong> Calculate precision, recall, and
          F1 score for each class separately.
        </li>
        <li>
          <strong>Balanced Accuracy:</strong> The average of recall obtained on
          each class.
        </li>
      </ul>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Per-class Metrics
print(classification_report(y_true, y_pred))

# Balanced Accuracy
balanced_acc = balanced_accuracy_score(y_true, y_pred)
  `}
      />

      <h3>4. Ranking Metrics</h3>
      <p>
        Ranking problems involve ordering a set of items based on their
        relevance or importance. These are common in information retrieval,
        recommendation systems, and search engines.
      </p>

      <h4>Mean Average Precision (MAP)</h4>
      <p>
        MAP measures the quality of rankings across multiple queries. It's the
        mean of the Average Precision scores for each query.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import average_precision_score

map_score = average_precision_score(y_true, y_score)
  `}
      />
      <p>
        <strong>Characteristic:</strong> MAP is sensitive to the entire ranking
        and provides a single score that summarizes the quality of rankings
        across all queries. However, it assumes binary relevance (an item is
        either relevant or not).
      </p>

      <h4>Normalized Discounted Cumulative Gain (NDCG)</h4>
      <p>
        NDCG measures the quality of rankings with emphasis on the top-ranked
        items. It takes into account the position of correct items in the ranked
        list.
      </p>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import ndcg_score

ndcg = ndcg_score(y_true, y_score)
  `}
      />
      <p>
        <strong>Characteristic:</strong> NDCG is particularly useful when you
        have graded relevance scores and when the order of items matters (e.g.,
        in search results). It penalizes highly relevant documents appearing
        lower in the ranking.
      </p>

      <h3>5. Time Series Metrics</h3>
      <p>
        Time series forecasting involves predicting future values based on
        historical data. These metrics are crucial in fields like finance,
        weather forecasting, and demand prediction.
      </p>

      <h4>Mean Absolute Percentage Error (MAPE)</h4>
      <p>
        MAPE measures the average percentage difference between predicted and
        actual values. It's scale-independent, making it useful for comparing
        forecasts across different scales.
      </p>
      <BlockMath math="MAPE = \frac{1}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right| \cdot 100" />
      <CodeBlock
        language="python"
        code={`
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_score = mape(y_true, y_pred)
  `}
      />
      <p>
        <strong>Characteristic:</strong> MAPE is intuitive and easy to interpret
        as it provides a percentage error. However, it can be problematic when
        actual values are close to zero and it's not defined for actual values
        of zero.
      </p>

      <h4>Other Time Series Metrics</h4>
      <p>
        Depending on the specific requirements of your time series forecasting
        task, you might also consider:
      </p>
      <ul>
        <li>
          <strong>Mean Absolute Error (MAE):</strong> Average of absolute
          errors, useful when the scale of errors matters.
        </li>
        <li>
          <strong>Root Mean Squared Error (RMSE):</strong> Square root of the
          average of squared errors, sensitive to large errors.
        </li>
        <li>
          <strong>Mean Absolute Scaled Error (MASE):</strong> Compares the
          forecast errors of the model in question to the errors of a naive
          forecast.
        </li>
      </ul>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

def mase(y_true, y_pred, y_train, seasonality):
    n = len(y_true)
    d = np.abs(np.diff(y_train, n=seasonality)).sum() / (n - seasonality)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

mase_score = mase(y_true, y_pred, y_train, seasonality=12)  # Example for monthly data
  `}
      />
      <p>
        When working with time series data, it's important to consider the
        specific characteristics of your data, such as seasonality, trends, and
        the presence of outliers, when choosing and interpreting these metrics.
      </p>

      <p>
        It's often beneficial to use multiple metrics to get a comprehensive
        view of model performance. Always consider the specific context of your
        problem when interpreting these metrics.
      </p>
    </Container>
  );
};

export default EvaluationMetrics;
