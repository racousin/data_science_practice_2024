import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { InlineMath, BlockMath } from "react-katex";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const FeatureSelectionAndDimensionalityReduction = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_feature_selection_and_dimensionality_reduction.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/feature_selection_and_dimensionality_reduction.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/feature_selection_and_dimensionality_reduction.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/feature_selection_and_dimensionality_reduction.ipynb";

  const metadata = {
    description:
      "This dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.",
    source: "Breast Cancer Wisconsin (Diagnostic) Data Set",
    target: "target (1 = malignant, 0 = benign)",
    listData: [
      {
        name: "mean radius",
        description: "Mean of distances from center to points on the perimeter",
      },
      {
        name: "mean texture",
        description: "Standard deviation of gray-scale values",
      },
      {
        name: "mean perimeter",
        description: "Mean size of the core tumor",
      },
      {
        name: "mean area",
        description: "Mean area of the core tumor",
      },
      {
        name: "mean smoothness",
        description: "Mean of local variation in radius lengths",
      },
      {
        name: "mean compactness",
        description: "Mean of perimeter^2 / area - 1.0",
      },
      {
        name: "mean concavity",
        description: "Mean of severity of concave portions of the contour",
      },
      {
        name: "mean concave points",
        description: "Mean for number of concave portions of the contour",
      },
      {
        name: "mean symmetry",
        description: "Mean symmetry of the cell nucleus",
      },
      {
        name: "mean fractal dimension",
        description: "Mean for 'coastline approximation' - 1",
      },
      {
        name: "radius error",
        description:
          "Standard error for the mean of distances from center to points on the perimeter",
      },
      {
        name: "texture error",
        description:
          "Standard error for standard deviation of gray-scale values",
      },
      {
        name: "perimeter error",
        description: "Standard error for mean size of the core tumor",
      },
      {
        name: "area error",
        description: "Standard error for mean area of the core tumor",
      },
      {
        name: "smoothness error",
        description:
          "Standard error for mean of local variation in radius lengths",
      },
      {
        name: "compactness error",
        description: "Standard error for mean of perimeter^2 / area - 1.0",
      },
      {
        name: "concavity error",
        description:
          "Standard error for mean of severity of concave portions of the contour",
      },
      {
        name: "concave points error",
        description:
          "Standard error for mean for number of concave portions of the contour",
      },
      {
        name: "symmetry error",
        description: "Standard error for mean symmetry of the cell nucleus",
      },
      {
        name: "fractal dimension error",
        description:
          "Standard error for mean for 'coastline approximation' - 1",
      },
      {
        name: "worst radius",
        description:
          "Worst or largest mean value for distance from center to points on the perimeter",
      },
      {
        name: "worst texture",
        description:
          "Worst or largest mean value for standard deviation of gray-scale values",
      },
      {
        name: "worst perimeter",
        description: "Worst or largest mean value for core tumor size",
      },
      {
        name: "worst area",
        description: "Worst or largest mean value for core tumor area",
      },
      {
        name: "worst smoothness",
        description:
          "Worst or largest mean value for local variation in radius lengths",
      },
      {
        name: "worst compactness",
        description: "Worst or largest mean value for perimeter^2 / area - 1.0",
      },
      {
        name: "worst concavity",
        description:
          "Worst or largest mean value for severity of concave portions of the contour",
      },
      {
        name: "worst concave points",
        description:
          "Worst or largest mean value for number of concave portions of the contour",
      },
      {
        name: "worst symmetry",
        description:
          "Worst or largest mean value for symmetry of the cell nucleus",
      },
      {
        name: "worst fractal dimension",
        description:
          "Worst or largest mean value for 'coastline approximation' - 1",
      },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">Feature Selection and Dimensionality Reduction</h1>

      <section>
        <h2 id="Selection">Feature Selection Techniques</h2>

        <p>
          Feature selection is the process of selecting a subset of relevant
          features for use in model construction. It's used to simplify models,
          reduce training times, and improve generalization by reducing
          overfitting.
        </p>

        <h3>1. Filter Methods</h3>
        <p>
          Filter methods select features based on their scores in various
          statistical tests for their correlation with the outcome variable.
        </p>

        <h4>Characteristics:</h4>
        <ul>
          <li>Independent of the learning algorithm</li>
          <li>Computationally simple and fast</li>
          <li>
            Considers the features separately (may miss feature interactions)
          </li>
        </ul>

        <h4>Examples:</h4>

        <h5>a) Correlation Coefficient</h5>
        <p>Measures the linear relationship between two variables.</p>
        <BlockMath math="r_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}" />

        <h5>b) Chi-Squared Test</h5>
        <p>Measures the dependence between categorical variables.</p>
        <BlockMath math="\chi^2 = \sum_{i=1}^n \frac{(O_i - E_i)^2}{E_i}" />

        <h4>Python Example:</h4>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2

# Assuming X is your feature matrix and y is your target variable
selector = SelectKBest(chi2, k=5)  # Select top 5 features
X_new = selector.fit_transform(X, y)

# Get the indices of the selected features
selected_features = selector.get_support(indices=True)
feature_names = X.columns[selected_features]
print("Selected features:", feature_names)
    `}
        />

        <h3>2. Wrapper Methods</h3>
        <p>
          Wrapper methods use a predictive model to score feature subsets. They
          train a new model for each subset and are computationally intensive.
        </p>

        <h4>Characteristics:</h4>
        <ul>
          <li>Interact with the learning algorithm</li>
          <li>More computationally intensive than filter methods</li>
          <li>
            Usually provide the best performing feature set for that particular
            type of model
          </li>
        </ul>

        <h4>Examples:</h4>

        <h5>a) Recursive Feature Elimination (RFE)</h5>
        <p>
          Recursively removes attributes and builds a model on those attributes
          that remain.
        </p>

        <h5>b) Forward Feature Selection</h5>
        <p>
          Iteratively adds the best feature to the set of selected features.
        </p>

        <h4>Python Example (RFE):</h4>
        <CodeBlock
          language="python"
          code={`
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Assuming X is your feature matrix and y is your target variable
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
fit = rfe.fit(X, y)

print("Selected features:", X.columns[fit.support_])
    `}
        />

        <h3>3. Embedded Methods</h3>
        <p>
          Embedded methods perform feature selection as part of the model
          creation process.
        </p>

        <h4>Characteristics:</h4>
        <ul>
          <li>Feature selection is built into the model training process</li>
          <li>More efficient than wrapper methods</li>
          <li>Capture feature interactions</li>
        </ul>

        <h4>Examples:</h4>

        <h5>a) Lasso Regression</h5>
        <p>
          Uses L1 regularization to shrink some coefficients to zero,
          effectively selecting features.
        </p>
        <BlockMath math="\min_{\beta} \frac{1}{n} \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2 + \lambda \sum_{j=1}^p |\beta_j|" />

        <h5>b) Random Forest Feature Importance</h5>
        <p>
          Uses the tree-based structure to rank features by their importance.
        </p>

        <h4>Python Example (Random Forest Feature Importance):</h4>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Assuming X is your feature matrix and y is your target variable
model = RandomForestClassifier()
model.fit(X, y)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
    `}
        />
      </section>

      <section>
        <h2 id="Reduction">Dimensionality Reduction Techniques</h2>

        <p>
          Dimensionality reduction techniques transform the data from a
          high-dimensional space into a lower-dimensional space, retaining
          meaningful properties of the original data. These methods can help
          with visualization, noise reduction, and improved computational
          efficiency.
        </p>

        <h3>1. Principal Component Analysis (PCA)</h3>
        <p>
          PCA is an unsupervised linear transformation technique that seeks to
          find the directions of maximum variance in high-dimensional data and
          project it onto a new subspace with equal or fewer dimensions than the
          original.
        </p>

        <h4>Characteristics:</h4>
        <ul>
          <li>Preserves global structure</li>
          <li>Maximizes variance and minimizes reconstruction error</li>
          <li>Computationally efficient</li>
          <li>May not be effective for non-linear relationships in data</li>
        </ul>

        <h4>Mathematical Formulation:</h4>
        <p>PCA finds the eigenvectors of the covariance matrix:</p>
        <BlockMath math="\Sigma = \frac{1}{n-1} X^T X" />
        <p>
          where X is the centered data matrix. The principal components are the
          eigenvectors corresponding to the largest eigenvalues.
        </p>

        <h4>Python Example:</h4>
        <CodeBlock
          language="python"
          code={`
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming X is your feature matrix
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)  # Assuming y is your target variable
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Dataset')
plt.colorbar()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
    `}
        />

        <h3>2. t-Distributed Stochastic Neighbor Embedding (t-SNE)</h3>
        <p>
          t-SNE is a nonlinear technique for dimensionality reduction that is
          particularly well suited for the visualization of high-dimensional
          datasets.
        </p>

        <h4>Characteristics:</h4>
        <ul>
          <li>Preserves local structure</li>
          <li>Particularly effective for visualization in 2D or 3D</li>
          <li>Computationally intensive</li>
          <li>
            Results can vary with different initializations and parameters
          </li>
        </ul>

        <h4>Mathematical Formulation:</h4>
        <p>
          t-SNE minimizes the Kullback-Leibler divergence between two
          distributions:
        </p>
        <BlockMath math="KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}" />
        <p>
          where <BlockMath math="p_{ij}" /> represents the similarity between
          points in high-dimensional space and <BlockMath math="q_{ij}" /> in
          low-dimensional space.
        </p>

        <h4>Python Example:</h4>
        <CodeBlock
          language="python"
          code={`
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming X is your feature matrix
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)  # Assuming y is your target variable
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE visualization of dataset')
plt.colorbar()
plt.show()
    `}
        />

        <h3>3. Linear Discriminant Analysis (LDA)</h3>
        <p>
          LDA is a supervised method used for both classification and
          dimensionality reduction. It projects the data onto a
          lower-dimensional space while maximizing the separation between
          classes.
        </p>

        <h4>Characteristics:</h4>
        <ul>
          <li>Supervised technique (uses class labels)</li>
          <li>Maximizes class separability</li>
          <li>
            Can be used for both dimensionality reduction and classification
          </li>
          <li>
            Assumes classes are normally distributed with equal covariance
          </li>
        </ul>

        <h4>Mathematical Formulation:</h4>
        <p>
          LDA maximizes the ratio of between-class variance to within-class
          variance:
        </p>
        <BlockMath math="J(w) = \frac{w^T S_B w}{w^T S_W w}" />
        <p>
          where S_B is the between-class scatter matrix and S_W is the
          within-class scatter matrix.
        </p>

        <h4>Python Example:</h4>
        <CodeBlock
          language="python"
          code={`
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Assuming X is your feature matrix and y is your target variable
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel('First LDA component')
plt.ylabel('Second LDA component')
plt.title('LDA of dataset')
plt.colorbar()
plt.show()

print("Explained variance ratio:", lda.explained_variance_ratio_)
    `}
        />
      </section>

      <section>
        <h2>Comparison and Best Practices</h2>

        <h3>
          Comparison of Feature Selection and Dimensionality Reduction
          Techniques
        </h3>

        <table className="table table-bordered">
          <thead>
            <tr>
              <th>Technique</th>
              <th>Pros</th>
              <th>Cons</th>
              <th>Best Use Cases</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Filter Methods</td>
              <td>
                <ul>
                  <li>Fast and computationally efficient</li>
                  <li>Independent of the learning algorithm</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>May not capture feature interactions</li>
                  <li>May select redundant features</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Large datasets with many features</li>
                  <li>Quick initial feature screening</li>
                </ul>
              </td>
            </tr>
            <tr>
              <td>Wrapper Methods</td>
              <td>
                <ul>
                  <li>Consider feature interactions</li>
                  <li>Can be tailored to specific learning algorithms</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Computationally intensive</li>
                  <li>Risk of overfitting</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Small to medium-sized datasets</li>
                  <li>When computational resources are not a constraint</li>
                </ul>
              </td>
            </tr>
            <tr>
              <td>Embedded Methods</td>
              <td>
                <ul>
                  <li>Consider feature interactions</li>
                  <li>More efficient than wrapper methods</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Specific to certain learning algorithms</li>
                  <li>Can be computationally intensive</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>
                    When using algorithms that support embedded feature
                    selection (e.g., Random Forests, Lasso)
                  </li>
                </ul>
              </td>
            </tr>
            <tr>
              <td>PCA</td>
              <td>
                <ul>
                  <li>Preserves global structure</li>
                  <li>Computationally efficient</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Reduced interpretability</li>
                  <li>Assumes linear relationships</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Datasets with many correlated features</li>
                  <li>When global variance is important</li>
                </ul>
              </td>
            </tr>
            <tr>
              <td>t-SNE</td>
              <td>
                <ul>
                  <li>Preserves local structure</li>
                  <li>Effective for visualization</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Computationally intensive</li>
                  <li>Results can vary with different runs</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>High-dimensional data visualization</li>
                  <li>Exploring local structure in data</li>
                </ul>
              </td>
            </tr>
            <tr>
              <td>LDA</td>
              <td>
                <ul>
                  <li>Maximizes class separability</li>
                  <li>Can be used for classification</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Assumes normal distribution with equal covariance</li>
                  <li>Limited to C-1 dimensions (C = number of classes)</li>
                </ul>
              </td>
              <td>
                <ul>
                  <li>Multi-class classification problems</li>
                  <li>When class separation is important</li>
                </ul>
              </td>
            </tr>
          </tbody>
        </table>

        <h3>Best Practices</h3>

        <ol>
          <li>
            <strong>Understand Your Data:</strong> Before applying any feature
            selection or dimensionality reduction technique, thoroughly explore
            and understand your dataset. This includes checking for
            correlations, distributions, and potential outliers.
          </li>

          <li>
            <strong>Consider the Problem Type:</strong> Choose techniques that
            are appropriate for your problem type (classification, regression,
            clustering) and data characteristics (linear/non-linear
            relationships, high/low dimensionality).
          </li>

          <li>
            <strong>Use Cross-Validation:</strong> When using wrapper or
            embedded methods, always use cross-validation to avoid overfitting.
          </li>

          <li>
            <strong>Combine Multiple Techniques:</strong> Often, a combination
            of techniques can yield better results. For example, you might use a
            filter method for initial screening, followed by a wrapper method
            for final selection.
          </li>

          <li>
            <strong>Consider Interpretability:</strong> If model
            interpretability is important, favor feature selection over
            dimensionality reduction, and consider methods that provide feature
            importance scores.
          </li>

          <li>
            <strong>Scale Your Data:</strong> Many dimensionality reduction
            techniques are sensitive to the scale of the input features. Always
            normalize or standardize your data before applying these techniques.
          </li>

          <li>
            <strong>Validate Results:</strong> After applying feature selection
            or dimensionality reduction, validate that the reduced feature set
            or transformed data still captures the important characteristics of
            your original dataset.
          </li>

          <li>
            <strong>Consider Computational Resources:</strong> For large
            datasets, start with computationally efficient methods (e.g., filter
            methods, PCA) before moving to more intensive techniques.
          </li>

          <li>
            <strong>Domain Knowledge is Key:</strong> Incorporate domain
            knowledge when possible. Sometimes, features that seem unimportant
            statistically may be crucial from a domain perspective.
          </li>

          <li>
            <strong>Iterative Process:</strong> Feature selection and
            dimensionality reduction are often iterative processes. Be prepared
            to try multiple techniques and combinations to find the best
            approach for your specific problem.
          </li>
        </ol>

        <h3>Visualization of Dimensionality Reduction Techniques</h3>

        <p>
          To illustrate the differences between PCA, t-SNE, and LDA, let's
          visualize their application on a sample dataset:
        </p>

        <CodeBlock
          language="python"
          code={`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_redundant=30, 
                           n_classes=3, n_clusters_per_class=1, random_state=42)

# Apply dimensionality reduction techniques
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42)
lda = LinearDiscriminantAnalysis(n_components=2)

X_pca = pca.fit_transform(X)
X_tsne = tsne.fit_transform(X)
X_lda = lda.fit_transform(X, y)

# Visualize results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
ax1.set_title('PCA')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
ax2.set_title('t-SNE')

ax3.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
ax3.set_title('LDA')

plt.tight_layout()
plt.show()
    `}
        />

        <p>
          This visualization demonstrates how different dimensionality reduction
          techniques can reveal different aspects of the data structure. PCA
          focuses on preserving global variance, t-SNE emphasizes local
          structure, and LDA maximizes class separation.
        </p>
      </section>
      <Row>
        <div id="notebook-example"></div>
        <DataInteractionPanel
          DataUrl={DataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
          metadata={metadata}
        />
      </Row>
    </Container>
  );
};

export default FeatureSelectionAndDimensionalityReduction;
