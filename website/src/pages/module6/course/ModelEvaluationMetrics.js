import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ModelEvaluationMetrics = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Model Evaluation Metrics</h1>
      <p>
        In this section, you will learn about various metrics for evaluating the
        performance of machine learning models.
      </p>
      <Row>
        <Col>
          <h2>Confusion Matrix, Accuracy, Precision, Recall, F1-score</h2>
          <p>
            These metrics are commonly used for evaluating the performance of
            classification models. The confusion matrix shows the number of true
            positives, true negatives, false positives, and false negatives.
            Accuracy measures the proportion of correct predictions. Precision
            measures the proportion of true positives among all positive
            predictions. Recall measures the proportion of true positives among
            all actual positives. The F1-score is the harmonic mean of precision
            and recall.
          </p>
          <CodeBlock
            code={`# Example of evaluating a classification model
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))`}
          />
          <h2>ROC Curve and AUC</h2>
          <p>
            The ROC curve (Receiver Operating Characteristic curve) is a
            graphical representation of the performance of a classification
            model. The AUC (Area Under the Curve) measures the area under the
            ROC curve and can be used as a summary metric for the model's
            performance.
          </p>
          <CodeBlock
            code={`# Example of plotting the ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_pred_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()`}
          />
          <h2>Cross-validation Techniques</h2>
          <p>
            Cross-validation techniques are used to estimate the performance of
            a model on unseen data. Examples include k-fold cross-validation and
            leave-one-out cross-validation.
          </p>
          <CodeBlock
            code={`# Example of k-fold cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ModelEvaluationMetrics;
