import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";
import "katex/dist/katex.min.css";

const SVM = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Support Vector Machines (SVM)</h1>

      <section>
        <h2 id="theory">The Kernel Trick</h2>
        <p>
          The main idea behind SVM is to find the optimal hyperplane that
          separates different classes in the feature space.
        </p>
        <p>
          For linearly separable data, SVM tries to maximize the margin, which
          is the distance between the hyperplane and the nearest data point from
          either class. The data points that lie on the margin are called
          support vectors.
        </p>
        <BlockMath math="f(x) = \text{sign}(\mathbf{w}^T\mathbf{x} + b)" />
        <p>
          Where <InlineMath math="\mathbf{w}" /> is the normal vector to the
          hyperplane and <InlineMath math="b" /> is the bias term.
        </p>
        <h3>Kernel Trick</h3>
        <p>
          The kernel trick allows SVM to operate in a high-dimensional feature
          space without explicitly computing the coordinates of the data in that
          space. This is done by using kernel functions that compute the inner
          product between two vectors in the feature space.
        </p>
        <p>Common kernel functions include:</p>
        <ul>
          <li>
            Linear:{" "}
            <InlineMath math="K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T\mathbf{y}" />
          </li>
          <li>
            Polynomial:{" "}
            <InlineMath math="K(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^T\mathbf{y} + r)^d" />
          </li>
          <li>
            Radial Basis Function (RBF):{" "}
            <InlineMath math="K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma ||\mathbf{x} - \mathbf{y}||^2)" />
          </li>
        </ul>
      </section>

      <section>
        <h2 id="types">Linear and Non-linear SVMs</h2>
        <h3>Linear SVM</h3>
        <p>
          Linear SVM uses a linear kernel and is suitable for linearly separable
          data or as a baseline model.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)
print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred_linear)}")
          `}
        />

        <h3>Non-linear SVM</h3>
        <p>
          Non-linear SVM uses kernels like RBF or polynomial to handle
          non-linearly separable data.
        </p>
        <CodeBlock
          language="python"
          code={`
# RBF Kernel SVM
rbf_svm = SVC(kernel='rbf', random_state=42)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)
print(f"RBF Kernel SVM Accuracy: {accuracy_score(y_test, y_pred_rbf)}")

# Polynomial Kernel SVM
poly_svm = SVC(kernel='poly', degree=3, random_state=42)
poly_svm.fit(X_train, y_train)
y_pred_poly = poly_svm.predict(X_test)
print(f"Polynomial Kernel SVM Accuracy: {accuracy_score(y_test, y_pred_poly)}")
          `}
        />
      </section>

      <section>
        {/* <h2 id="implementation">Implementation and Key Hyperparameters</h2>
        <p>
          SVMs in scikit-learn are implemented in the <code>SVC</code> (Support
          Vector Classification), <code>SVR</code> (Support Vector Regression),
          and <code>LinearSVC</code> (Linear Support Vector Classification)
          classes.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
print(f"Best SVM Accuracy: {accuracy_score(y_test, y_pred_best)}")
          `}
        /> */}

        <h3>Key Hyperparameters</h3>
        <ul>
          <li>
            <code>C</code>: Regularization parameter. Controls the trade-off
            between achieving a low training error and a low testing error.
          </li>
          <li>
            <code>kernel</code>: Specifies the kernel type to be used ('linear',
            'poly', 'rbf', 'sigmoid').
          </li>
          <li>
            <code>gamma</code>: Kernel coefficient for 'rbf', 'poly' and
            'sigmoid' kernels. Controls the influence of a single training
            example.
          </li>
          <li>
            <code>degree</code>: Degree of the polynomial kernel function.
          </li>
          <li>
            <code>class_weight</code>: Sets the parameter C of class i to
            class_weight[i]*C for SVC.
          </li>
        </ul>
      </section>

      {/* <section>
        <h2>Pros and Cons of SVM</h2>
        <h3>Pros</h3>
        <ul>
          <li>Effective in high-dimensional spaces</li>
          <li>
            Still effective when the number of dimensions is greater than the
            number of samples
          </li>
          <li>
            Uses a subset of training points (support vectors) in the decision
            function, making it memory efficient
          </li>
          <li>
            Different kernel functions can be specified for the decision
            function
          </li>
        </ul>
        <h3>Cons</h3>
        <ul>
          <li>
            If the number of features is much greater than the number of
            samples, avoiding over-fitting requires careful choice of kernel
            functions and regularization term
          </li>
          <li>
            Does not directly provide probability estimates (these are
            calculated using cross-validation, which can be slow)
          </li>
          <li>Can be computationally expensive for large datasets</li>
        </ul>
      </section> */}
    </Container>
  );
};

export default SVM;
