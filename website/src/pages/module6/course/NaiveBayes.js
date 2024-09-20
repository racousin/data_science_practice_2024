import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const NaiveBayes = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Naive Bayes</h1>

      <section>
        <h2 id="theory">Theory and Types</h2>
        <p>
          Naive Bayes is a family of probabilistic algorithms based on applying
          Bayes' theorem with the "naive" assumption of conditional independence
          between features given the class label.
        </p>
        <BlockMath math="P(y|x_1, ..., x_n) = \frac{P(y) \prod_{i=1}^n P(x_i|y)}{P(x_1, ..., x_n)}" />
        <p>
          Where <InlineMath math="y" /> is the class variable and{" "}
          <InlineMath math="x_1" /> through <InlineMath math="x_n" /> are the
          feature variables.
        </p>

        <h3>Types of Naive Bayes Classifiers</h3>
        <ol>
          <li>
            <strong>Gaussian Naive Bayes:</strong> Assumes that the features
            follow a normal distribution.
          </li>
          <li>
            <strong>Multinomial Naive Bayes:</strong> Suitable for discrete
            features (e.g., word counts for text classification).
          </li>
          <li>
            <strong>Bernoulli Naive Bayes:</strong> Useful for binary features.
          </li>
        </ol>
      </section>

      <section>
        <h2 id="implementation">Implementation and Use Cases</h2>
        <CodeBlock
          language="python"
          code={`
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Gaussian Naive Bayes
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_gnb))

# Multinomial Naive Bayes (for discrete features)
# We'll use the Iris dataset and convert it to integer counts
iris = load_iris()
X, y = iris.data, iris.target
X = np.round(X).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
print("Multinomial Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_mnb))

# Bernoulli Naive Bayes (for binary features)
X_binary = X > X.mean(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=42)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)
print("Bernoulli Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_bnb))

# Detailed classification report for Gaussian Naive Bayes
print("\nClassification Report for Gaussian Naive Bayes:")
print(classification_report(y_test, y_pred_gnb))
          `}
        />
        {/* 
        <h3>Use Cases</h3>
        <ul>
          <li>
            Text classification (e.g., spam detection, sentiment analysis)
          </li>
          <li>Document categorization</li>
          <li>Disease prediction</li>
          <li>Real-time prediction (due to its efficiency)</li>
          <li>Recommendation systems</li>
        </ul>
      </section>

      <section>
        <h2 id="pros-cons">Pros and Cons</h2>
        <h3>Pros</h3>
        <ul>
          <li>Simple and easy to implement</li>
          <li>Works well with high-dimensional data</li>
          <li>Requires relatively small amount of training data</li>
          <li>Fast for both training and prediction</li>
          <li>Not sensitive to irrelevant features</li>
          <li>
            Works well for both binary and multi-class classification problems
          </li>
        </ul>
        <h3>Cons</h3>
        <ul>
          <li>
            Assumes strong independence between features, which is often not the
            case in real-world scenarios
          </li>
          <li>Sensitive to feature correlation</li>
          <li>Cannot learn interactions between features</li>
          <li>
            May give poor estimates of probabilities (but often still gives good
            classifications)
          </li>
        </ul>
      </section>

      <section>
        <h2 id="feature-importance">Feature Importance in Naive Bayes</h2>
        <p>
          Although Naive Bayes doesn't provide feature importance scores
          directly like some other algorithms, we can derive feature importance
          from the learned parameters.
        </p>
        <CodeBlock
          language="python"
          code={`
import numpy as np
import matplotlib.pyplot as plt

# For Gaussian Naive Bayes
feature_importance = np.abs(gnb.theta_[1] - gnb.theta_[0])
feature_names = [f"Feature {i}" for i in range(20)]

plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.title("Feature Importance in Gaussian Naive Bayes")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
          `}
        />
      </section>

      <section>
        <h2>Conclusion</h2>
        <p>
          Naive Bayes classifiers are simple yet powerful algorithms that work
          well for a variety of classification tasks. Their simplicity,
          efficiency, and ability to handle high-dimensional data make them
          particularly useful for text classification problems and as baseline
          models. However, their "naive" assumption of feature independence can
          limit their performance in scenarios where feature interactions are
          important. As with any machine learning algorithm, it's important to
          understand the assumptions and limitations of Naive Bayes when
          applying it to real-world problems.
        </p>
      </section> */}
      </section>
    </Container>
  );
};

export default NaiveBayes;
