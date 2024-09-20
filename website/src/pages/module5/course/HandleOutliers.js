import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleOutliers = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const trainDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_outliers_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_outliers_test.csv";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/handling_outliers.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/handling_outliers.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/handling_outliers.ipynb";
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
      <h1 className="my-4">Handling Outliers</h1>

      <Row>
        <Col>
          <h2 id="what-are-outliers">What Are Outliers?</h2>
          <p>
            Outliers are data points that differ significantly from other
            observations. They can arise due to variability in the measurement
            or experimental errors, and can sometimes be indicative of
            fraudulent behavior.
          </p>

          <h2 id="detecting-outliers">Detecting Outliers</h2>
          <p>
            Detecting outliers can be done using statistical methods or
            visualizations:
          </p>
          <ul>
            <li>
              <strong>Standard Deviation:</strong> Points that are more than 3
              standard deviations from the mean are often considered outliers.
            </li>
            <li>
              <strong>Interquartile Range (IQR):</strong> Data points that fall
              below Q1 - 1.5xIQR or above Q3 + 1.5xIQR are typically considered
              outliers.
            </li>
            <li>
              <strong>Box Plots:</strong> Visual method for identifying
              outliers, showing data distribution through quartiles.
            </li>
            <li>
              <strong>Scatter Plots:</strong> Help in spotting outliers in the
              context of how data points are clustered or spread out.
            </li>
          </ul>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd
import numpy as np

# Example using IQR
Q1 = df['data'].quantile(0.25)
Q3 = df['data'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['data'] < Q1 - 1.5 * IQR) | (df['data'] > Q3 + 1.5 * IQR)]
print(outliers)`}
          />
          <h2 id="visualize-outliers">Visualize Outliers</h2>
          <p>
            Visualizing outliers is an effective way to identify and understand
            the anomalies in your data. Visualization helps in determining
            whether outliers are the result of data entry errors, experimental
            errors, or natural variation in the dataset.
          </p>
          <CodeBlock
            language={"python"}
            code={`import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df' is your DataFrame and 'data' is your column of interest
# Boxplot to visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['data'])
plt.title('Box Plot for Outlier Detection')
plt.show()

# Scatter plot for multidimensional outlier analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='data', y='another_feature', data=df)
plt.title('Scatter Plot for Outlier Detection')
plt.xlabel('Data')
plt.ylabel('Another Feature')
plt.show()`}
          />

          <p>
            The box plot provides a clear visual representation of the
            distribution of the data, highlighting outliers as points that
            appear far from the rest of the data. Scatter plots are useful for
            detecting outliers in a multi-dimensional space, helping to identify
            patterns or groups of outliers that may not be apparent in
            one-dimensional plots.
          </p>

          <h2 id="managing-outliers">Managing Outliers</h2>
          <p>
            Depending on the nature and the source of the outliers, you may
            choose to remove them or adjust them:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Removing outliers
filtered_df = df[~((df['data'] < (Q1 - 1.5 * IQR)) | (df['data'] > (Q3 + 1.5 * IQR)))]

# Adjusting outliers by capping
df.loc[df['data'] < (Q1 - 1.5 * IQR), 'data'] = Q1 - 1.5 * IQR
df.loc[df['data'] > (Q3 + 1.5 * IQR), 'data'] = Q3 + 1.5 * IQR`}
          />

          <h2 id="considerations">Considerations</h2>
          <p>
            It's important to understand the impact of modifying outliers within
            your dataset. Removing or adjusting outliers without proper analysis
            can lead to biased results, especially in small datasets or datasets
            with natural variability.
          </p>
        </Col>
      </Row>
      <Row>
        <div id="notebook-example"></div>
        <DataInteractionPanel
          trainDataUrl={trainDataUrl}
          testDataUrl={testDataUrl}
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

export default HandleOutliers;
