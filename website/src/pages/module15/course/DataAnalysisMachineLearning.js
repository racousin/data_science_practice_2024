import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DataAnalysisMachineLearning = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Data Analysis and Machine Learning on GCP</h1>
      <p>
        In this section, you will learn how to utilize GCP services for data
        analysis and deploying machine learning models.
      </p>
      <Row>
        <Col>
          <h2>Using BigQuery for Large-scale Data Analysis</h2>
          <p>
            BigQuery is a fully managed, serverless data warehouse that can
            handle large amounts of data. It supports SQL and can be used for
            ad-hoc queries, batch processing, and machine learning.
          </p>
          <h2>Machine Learning Services: AI Platform, AutoML</h2>
          <p>
            AI Platform is a service for building, training, and deploying
            machine learning models. AutoML is a service for building machine
            learning models with minimal coding.
          </p>
          <h2>Building and Deploying Models with TensorFlow on AI Platform</h2>
          <p>
            TensorFlow is an open-source machine learning framework that can be
            used to build and train models. AI Platform provides a managed
            service for training and deploying TensorFlow models.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default DataAnalysisMachineLearning;
