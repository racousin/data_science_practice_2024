import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Exercise2 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">
        Exercise 2: Building a Machine Learning Baseline Pipeline
      </h1>
      <p>
        In this exercise, you will develop a simple machine learning pipeline
        using scikit-learn to model the data explored in Exercise 1.
      </p>
      <Row>
        <Col>
          <h2>Requirements</h2>
          <h3>Feature Selection and Preprocessing</h3>
          <ul>
            <li>
              Select features to be used for modeling based on the EDA findings.
            </li>
            <li>Apply appropriate preprocessing steps (scaling, encoding).</li>
          </ul>
          <h3>Model Selection</h3>
          <ul>
            <li>
              Choose a simple model as a baseline (e.g., Logistic Regression,
              Decision Tree, or Linear Regression).
            </li>
            <li>
              Explain why this model is chosen based on the characteristics of
              the data.
            </li>
          </ul>
          <h3>Model Training and Evaluation</h3>
          <ul>
            <li>Split the data into training and test sets.</li>
            <li>Train the model on the training data.</li>
            <li>
              Evaluate the model on the test data using appropriate metrics
              (accuracy, RMSE, etc.).
            </li>
            <li>
              Display the confusion matrix or regression plots, depending on the
              type of problem.
            </li>
          </ul>
          <h3>Baseline Results</h3>
          <ul>
            <li>Discuss the baseline model’s performance.</li>
            <li>
              Suggest potential improvements or next steps for enhancing the
              model.
            </li>
          </ul>
          <h3>Deliverables</h3>
          <ul>
            <li>
              Jupyter Notebook for Exercise 1: This should include all codes,
              plots, and analyses along with descriptive text explaining each
              step.
            </li>
            <li>
              Jupyter Notebook for Exercise 2: This should detail the process of
              setting up the machine learning pipeline, including code and
              commentary on each decision made in the process.
            </li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise2;
