import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const AdvancedTechniques = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Advanced Feature Engineering Techniques</h1>
      <p>
        In this section, you will learn about advanced strategies for feature
        engineering.
      </p>
      <Row>
        <Col>
          <h2>Dealing with Imbalanced Data</h2>
          <p>
            Imbalanced data can be handled using techniques such as
            oversampling, undersampling, and synthetic data generation.
          </p>
          <CodeBlock
            language={"python"}
            code={`# Example of oversampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)`}
          />
          <h2>Generating Synthetic Features</h2>
          <p>
            Synthetic features can be generated using techniques such as
            polynomial features and interaction terms.
          </p>
          <CodeBlock
            language={"python"}
            code={`# Example of generating synthetic features
X['interaction'] = X['feature1'] * X['feature2']`}
          />
          <h2>Time-series Specific Transformations</h2>
          <p>
            Time-series data can be transformed using techniques such as window
            functions and lag features.
          </p>
          <CodeBlock
            language={"python"}
            code={`# Example of lag features
X['lag1'] = X['feature'].shift(1)
X['lag2'] = X['feature'].shift(2)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default AdvancedTechniques;
