import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import DataInteractionPanel from "components/DataInteractionPanel";
import CodeBlock from "components/CodeBlock";

const Exercise1 = () => {
  const trainDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module6/exercise/module6_exercise_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module6/exercise/module6_exercise_test.csv";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module6/exercise/module6_exercise.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module6/exercise/module6_exercise.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module6/exercise/module6_exercise.ipynb";
  const metadata = {
    description:
      "This dataset includes various property metrics crucial for analyzing real estate sales, features, and pricing.",
    source: "Real Estate Transaction Records",
    target: "SalePrice",
    listData: [
      {
        name: "BedroomAbvGr",
        description:
          "Bedrooms above grade (does NOT include basement bedrooms)",
      },
      { name: "KitchenAbvGr", description: "Kitchens above grade" },
      {
        name: "TotRmsAbvGrd",
        description: "Total rooms above grade (does not include bathrooms)",
      },
      { name: "Fireplaces", description: "Number of fireplaces" },
      { name: "GarageYrBlt", description: "Year garage was built" },
      { name: "GarageCars", description: "Size of garage in car capacity" },
      { name: "GarageArea", description: "Size of garage in square feet" },
      { name: "WoodDeckSF", description: "Wood deck area in square feet" },
      { name: "OpenPorchSF", description: "Open porch area in square feet" },
      {
        name: "EnclosedPorch",
        description: "Enclosed porch area in square feet",
      },
      {
        name: "3SsnPorch",
        description: "Three season porch area in square feet",
      },
      { name: "ScreenPorch", description: "Screen porch area in square feet" },
      { name: "PoolArea", description: "Pool area in square feet" },
      { name: "MiscVal", description: "Value of miscellaneous feature" },
      { name: "MoSold", description: "Month Sold (MM)" },
      { name: "YrSold", description: "Year Sold (YYYY)" },
      { name: "SalePrice", description: "Price of sale (target variable)" },
    ],
  };

  return (
    <Container fluid>
      <h1 className="my-4">
        Exercise 1: Data Exploration and Baseline Prediction
      </h1>
      <p>
        In this exercise, you will perform exploratory data analysis (EDA) and
        generate a baseline prediction. This task integrates data understanding
        with predictive modeling.
      </p>
      <Row>
        <Col>
          <h2>Overview</h2>
          <ul>
            <li>Load the dataset and explore its structure and statistics.</li>
            <li>
              Create visualizations to understand the data distributions and
              relationships.
            </li>
            <li>
              Develop a baseline model to make predictions on the test dataset.
            </li>
            <li>
              Explain your data discovery and analytical approach when
              presenting your notebook to a reviewer.
            </li>
            <li>
              Your submission will be reviewed, and feedback provided once you
              merge your pull request to the main branch.
            </li>
          </ul>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Expected Output</h2>
          <p>
            You are expected to submit a Jupyter Notebook containing the EDA and
            baseline model predictions. The notebook should include:
          </p>
          <ul>
            <li>Detailed annotations explaining your analysis and findings.</li>
            <li>
              Visualizations supporting the insights derived from the data.
            </li>
            <li>
              Predictions output saved in a CSV format, meeting the specified
              requirements.
            </li>
          </ul>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Evaluation</h2>
          <ol>
            <li>
              Create a CSV file named <code>submission.csv</code> with two
              columns:
              <ul>
                <li>
                  <code>id</code>: The identifier for each prediction.
                </li>
                <li>
                  <code>SalePrice</code>: The predicted values.
                </li>
              </ul>
            </li>
            <CodeBlock
              code={`id,SalePrice\n1,200000\n2,250000\n3,300000\n...`}
            />
            <li>
              Save the <code>submission.csv</code> file in the{" "}
              <code>module6</code> directory under your username folder.
            </li>
            <li>
              Ensure your predictions file is in the correct format and contains
              the required columns.
            </li>
          </ol>
          <p>
            Your predictions will be evaluated using the Mean Absolute Error
            (MAE) metric. The error threshold for this exercise is 34000. Ensure
            your predictions are accurate enough to meet this threshold.
          </p>
        </Col>
      </Row>
      <Row>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </Row>
    </Container>
  );
};

export default Exercise1;
