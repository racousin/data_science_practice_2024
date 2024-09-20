import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BestPracticesAndResources = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Best Practices and Resources for Colab</h1>

      <Row>
        <Col md={12}>
          <h2 id="installing-packages-in-colab">
            Installing Packages in Colab
          </h2>
          <p>
            Google Colab supports most libraries used in data science and
            machine learning, but sometimes you may need to install additional
            packages. Here’s how you can do it:
          </p>
          <CodeBlock code={`!pip install package_name`} />

          <h3>Example: Installing a Package from GitHub</h3>
          <p>To install a Python package from a GitHub repository:</p>
          <CodeBlock
            code={`!pip install git+https://github.com/username/repository.git`}
          />

          <h3>Specific Branch or Tag</h3>
          <p>To install from a specific branch or tag:</p>
          <CodeBlock
            code={`!pip install git+https://github.com/username/repository.git@branch_name`}
          />

          <h3>Installing a Package Located in a Subdirectory</h3>
          <p>
            If the package you want to install is in a subdirectory of a
            repository:
          </p>
          <CodeBlock
            code={`!pip install git+https://github.com/username/repository.git#subdirectory=folder_path`}
          />
        </Col>
      </Row>

      <Row className="mt-4">
        <Col md={12}>
          <h2 id="using-gpu-in-colab">
            Using GPU and Other Resources in Colab
          </h2>
          <p>
            Google Colab offers free access to GPUs and TPUs which can be
            incredibly beneficial for training machine learning models. Here’s
            how to set them up:
          </p>

          <h3>Activating a GPU</h3>
          <p>
            To change the runtime type and select GPU as your hardware
            accelerator:
          </p>
          <ul>
            <li>Go to the "Runtime" menu.</li>
            <li>Select "Change runtime type."</li>
            <li>
              In the dialog that appears, select GPU from the Hardware
              accelerator drop-down list.
            </li>
            <li>Click Save.</li>
          </ul>

          <h3>Activating a TPU</h3>
          <p>To use a TPU:</p>
          <ul>
            <li>
              Follow the same steps as for the GPU but select TPU from the
              Hardware accelerator list.
            </li>
          </ul>
        </Col>
      </Row>
      <Row>
        <Col md={12}>
          <h2 id="bi-tools">BI Tools</h2>
          <p>
            Business Intelligence tools play a critical role in Exploratory Data
            Analysis, providing interactive, visual ways to explore, analyze,
            and visualize data. These tools help identify patterns, trends, and
            relationships in data that are not readily apparent.
          </p>

          <h3>Popular BI Tools for Data Analysis</h3>
          <p>Some of the most widely used BI tools include:</p>
          <ul>
            <li>
              <strong>Tableau:</strong> Known for its powerful drag-and-drop
              capabilities that enable users to create complex visualizations
              without programming knowledge.
            </li>
            <li>
              <strong>Power BI:</strong> A Microsoft product that integrates
              seamlessly with other Microsoft services like Azure and Office
              365, offering comprehensive analytics capabilities.
            </li>
            <li>
              <strong>Qlik:</strong> Offers associative analytics that highlight
              the relationships between data points, promoting a deeper
              understanding of the data.
            </li>
          </ul>

          <h3>Getting Started with BI Tools</h3>
          <p>Here’s how to start with each tool:</p>
          <ol>
            <li>
              <strong>Tableau:</strong> Download the Tableau Desktop application
              or use Tableau Public for free.
            </li>
            <li>
              <strong>Power BI:</strong> Use Power BI Desktop for free or
              integrate it into the Office 365 suite for more features.
            </li>
            <li>
              <strong>Qlik Sense:</strong> Access Qlik’s cloud services or
              download the desktop application to explore its features.
            </li>
          </ol>

          <h3>Resources and Learning Materials</h3>
          <p>
            To help you get started with these BI tools, here are some helpful
            resources:
          </p>
          <ul>
            <li>
              <a
                href="https://www.tableau.com/learn/training"
                target="_blank"
                rel="noopener noreferrer"
              >
                Tableau Training and Tutorials
              </a>
            </li>
            <li>
              <a
                href="https://docs.microsoft.com/en-us/power-bi/fundamentals/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Power BI Documentation and Learning Path
              </a>
            </li>
            <li>
              <a
                href="https://help.qlik.com/en-US/sense/June2020/Content/Sense_Helpsites/Home-Sense.html"
                target="_blank"
                rel="noopener noreferrer"
              >
                Qlik Sense Help and Tutorials
              </a>
            </li>
          </ul>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col md={12}>
          <h3 id="resources">Useful Links and Resources</h3>
          <p>Here are some links to get more out of Google Colab:</p>
          <ul>
            <li>
              <a
                href="https://colab.research.google.com/notebooks/intro.ipynb"
                target="_blank"
                rel="noopener noreferrer"
              >
                Introduction to Colab
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/notebooks/basic_features_overview.ipynb"
                target="_blank"
                rel="noopener noreferrer"
              >
                Colab Features Overview
              </a>
            </li>
            <li>
              <a
                href="https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb"
                target="_blank"
                rel="noopener noreferrer"
              >
                Introduction to Pandas
              </a>
            </li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default BestPracticesAndResources;
