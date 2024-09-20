import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";

const ExploratoryDataAnalysis = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exploratory Data Analysis (EDA)</h1>

      <section>
        <h2>Definition and Importance</h2>
        <p>
          Exploratory Data Analysis (EDA) is an approach to analyzing datasets
          to summarize their main characteristics, often with visual methods.
          It's a critical step in the data science process that precedes formal
          modeling or hypothesis testing.
        </p>
        <p>
          <strong>Key Objectives:</strong>
        </p>
        <ul>
          <li>Understand the structure and characteristics of the data</li>
          <li>
            Identify patterns, trends, and relationships between variables
          </li>
          <li>Detect anomalies, outliers, and missing data</li>
          <li>Formulate hypotheses and guide further analysis</li>
        </ul>
      </section>
      <Row>
        <Col md={12}>
          <h2 id="jupyter-notebooks">Jupyter Notebooks</h2>
          <p>
            Jupyter Notebooks provide an interactive interface to combine
            executable code, rich text, and visualizations into a single
            document. They are extensively used in data science for exploratory
            analysis, data cleaning, statistical modeling, and visualization.
          </p>

          <h3>Installing Jupyter Notebooks</h3>
          <p>
            To get started with Jupyter Notebooks on your local machine, install
            it using pip:
          </p>
          <CodeBlock code={`pip install notebook`} />

          <h3>Launching Jupyter Notebooks</h3>
          <p>
            After installation, you can start the Jupyter Notebook by running:
          </p>
          <CodeBlock code={`jupyter notebook`} />
          <p>This command will open Jupyter in your default web browser.</p>
        </Col>
      </Row>

      <Row className="mt-4">
        <Col md={12}>
          <h2 id="google-colab">Google Colab</h2>
          <p>
            Google Colab is a cloud-based version of the Jupyter notebook
            designed for machine learning education and research. It provides a
            platform to write and execute arbitrary Python code through the
            browser, and is especially well-suited for machine learning, data
            analysis, and education.
          </p>

          <h3>Advantages of Google Colab</h3>
          <ul>
            <li>Free access to computing resources including GPUs and TPUs.</li>
            <li>No setup required to use Python libraries.</li>
            <li>Easy sharing and collaboration features.</li>
          </ul>

          <h3>Getting Started with Google Colab</h3>
          <p>
            To begin using Google Colab, navigate to the Colab website and start
            a new project:
          </p>
          <a
            href="https://colab.research.google.com/"
            target="_blank"
            rel="noopener noreferrer"
          >
            https://colab.research.google.com/
          </a>
        </Col>
      </Row>
      <section>
        <h2>Main Components of EDA</h2>

        <h3>1. Data Collection and Cleaning</h3>
        <p>
          Before analysis can begin, data must be collected and cleaned. This
          involves:
        </p>
        <ul>
          <li>Gathering data from various sources</li>
          <li>Handling missing values</li>
          <li>Correcting inconsistencies</li>
          <li>Dealing with outliers</li>
        </ul>

        <h3>2. Univariate Analysis</h3>
        <p>
          Examining variables individually to understand their distribution and
          characteristics.
        </p>
        <ul>
          <li>For categorical variables: frequency counts, bar charts</li>
          <li>
            For numerical variables: mean, median, mode, range, histograms, box
            plots
          </li>
        </ul>

        <h3>3. Bivariate Analysis</h3>
        <p>Exploring relationships between pairs of variables.</p>
        <ul>
          <li>Scatter plots for numerical-numerical relationships</li>
          <li>Box plots for numerical-categorical relationships</li>
          <li>Correlation coefficients</li>
        </ul>

        <h3>4. Multivariate Analysis</h3>
        <p>
          Analyzing relationships between three or more variables
          simultaneously.
        </p>
        <ul>
          <li>Pair plots</li>
          <li>Heat maps for correlation matrices</li>
          <li>Dimensionality reduction techniques (e.g., PCA)</li>
        </ul>

        <h3>5. Feature Engineering</h3>
        <p>
          Creating new features or transforming existing ones based on insights
          gained from the analysis.
        </p>
        <ul>
          <li>Combining variables</li>
          <li>Binning continuous variables</li>
          <li>Encoding categorical variables</li>
        </ul>
      </section>

      <section>
        <h2>Key Techniques and Tools</h2>
        <ul>
          <li>
            Summary Statistics: mean, median, standard deviation, percentiles
          </li>
          <li>
            Data Visualization: histograms, box plots, scatter plots, heat maps
          </li>
          <li>
            Correlation Analysis: Pearson correlation, Spearman rank correlation
          </li>
          <li>Dimensionality Reduction: Principal Component Analysis (PCA)</li>
        </ul>
      </section>

      <section>
        <h2>Importance in the Data Science Pipeline</h2>
        <p>EDA is crucial because it:</p>
        <ul>
          <li>Informs data preprocessing and feature engineering decisions</li>
          <li>Guides the selection of appropriate modeling techniques</li>
          <li>
            Helps in identifying potential issues early in the analysis process
          </li>
          <li>Provides context for interpreting model results</li>
        </ul>
      </section>

      <section>
        <h2>Best Practices</h2>
        <ul>
          <li>Start with simple visualizations and summary statistics</li>
          <li>Be systematic in your approach, examining all variables</li>
          <li>Look for patterns and anomalies in the data</li>
          <li>Document your findings and hypotheses</li>
          <li>
            Iterate between EDA and other stages of the data science process
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default ExploratoryDataAnalysis;
