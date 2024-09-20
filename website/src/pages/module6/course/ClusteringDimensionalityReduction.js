import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ClusteringDimensionalityReduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Clustering and Dimensionality Reduction</h1>
      <p>
        In this section, you will learn about unsupervised learning techniques
        used for grouping data and reducing dimensionality.
      </p>
      <Row>
        <Col>
          <h2>K-means Clustering</h2>
          <p>
            K-means clustering is a simple and widely used unsupervised learning
            technique that groups data points into a fixed number of clusters
            based on their similarity.
          </p>
          <CodeBlock
            code={`# Example of k-means clustering
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(X)
labels = model.labels_`}
          />
          <h2>Hierarchical Clustering</h2>
          <p>
            Hierarchical clustering is a type of unsupervised learning technique
            that creates a tree-like structure of clusters based on their
            similarity.
          </p>
          <CodeBlock
            code={`# Example of hierarchical clustering
from scipy.cluster.hierarchy import linkage, dendrogram

Z = linkage(X, 'ward')
dendrogram(Z)`}
          />
          <h2>PCA (Principal Component Analysis)</h2>
          <p>
            PCA is a dimensionality reduction technique that transforms
            high-dimensional data into a lower-dimensional representation while
            preserving as much of the original information as possible.
          </p>
          <CodeBlock
            code={`# Example of PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default ClusteringDimensionalityReduction;
