import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Exercise1 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">
        Exercise 1: Data Exploration with Jupyter Notebook
      </h1>
      <p>
        In this exercise, you will use a Jupyter Notebook to perform exploratory
        data analysis (EDA) on a provided dataset.
      </p>
      <Row>
        <Col>
          <h2>Requirements</h2>
          <h3>Load and Inspect the Data</h3>
          <ul>
            <li>
              Load the dataset into a DataFrame and display the first few rows.
            </li>
            <li>
              Describe the dataset, showing statistics, types, and missing
              values.
            </li>
          </ul>
          <h3>Visualizations</h3>
          <ul>
            <li>
              Generate histograms for all numerical features to understand
              distributions.
            </li>
            <li>Create box plots to identify outliers.</li>
            <li>
              Use scatter plots to explore relationships between features.
            </li>
          </ul>
          <h3>Data Cleaning (if needed based on the dataset)</h3>
          <ul>
            <li>Handle missing values by imputation or removal.</li>
            <li>
              Convert categorical data to numeric using encoding techniques.
            </li>
          </ul>
          <h3>Summary of Findings</h3>
          <ul>
            <li>
              Summarize key insights from the exploratory data analysis,
              including any potential issues, interesting correlations, or
              hypotheses about the data.
            </li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise1;
