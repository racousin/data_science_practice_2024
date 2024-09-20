import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleDuplicates = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const trainDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_duplicate_train.csv";
  const testDataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_duplicate_test.csv";
  const notebookUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/handling_duplicate.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/handling_duplicate.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/handling_duplicate.ipynb";
  const metadata = {
    description:
      "This dataset contains detailed meteorological measurements from various weather stations, capturing daily climatic conditions aimed at aiding weather forecasting and climatic research.",
    source: "National Weather Service",
    target: "Precipit", // Assuming Precipitation is a key measure to predict or analyze; adjust if different.
    listData: [
      {
        name: "id",
        description: "Unique identifier for each record.",
        dataType: "Integer",
        example: "1",
      },
      {
        name: "station_id",
        description: "Identifier for the weather station.",
        dataType: "Integer",
        example: "1004",
      },
      {
        name: "Date",
        description: "Date of observation.",
        dataType: "Date",
        example: "2018-02-26",
      },
      {
        name: "Temp_max",
        description:
          "Maximum temperature recorded on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "67.97°F",
      },
      {
        name: "Temp_avg",
        description: "Average temperature on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "60.39°F",
      },
      {
        name: "Temp_min",
        description:
          "Minimum temperature recorded on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "55.23°F",
      },
      {
        name: "Dew_max",
        description: "Maximum dew point on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "55.10°F",
      },
      {
        name: "Dew_avg",
        description: "Average dew point on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "59.39°F",
      },
      {
        name: "Dew_min",
        description: "Minimum dew point on the day, in degrees Fahrenheit.",
        dataType: "Continuous",
        example: "54.76°F",
      },
      {
        name: "Hum_max",
        description: "Maximum humidity recorded on the day, in percentage.",
        dataType: "Continuous",
        example: "96.85%",
      },
      {
        name: "Hum_avg",
        description: "Average humidity on the day, in percentage.",
        dataType: "Continuous",
        example: "80.60%",
      },
      {
        name: "Hum_min",
        description: "Minimum humidity on the day, in percentage.",
        dataType: "Continuous",
        example: "60.21%",
      },
      {
        name: "Wind_max",
        description:
          "Maximum wind speed recorded on the day, in miles per hour.",
        dataType: "Continuous",
        example: "12.94 mph",
      },
      {
        name: "Wind_avg",
        description: "Average wind speed on the day, in miles per hour.",
        dataType: "Continuous",
        example: "7.71 mph",
      },
      {
        name: "Wind_min",
        description: "Minimum wind speed on the day, in miles per hour.",
        dataType: "Continuous",
        example: "5.12 mph",
      },
      {
        name: "Press_max",
        description:
          "Maximum atmospheric pressure on the day, in inches of mercury.",
        dataType: "Continuous",
        example: "30.67 inHg",
      },
      {
        name: "Press_avg",
        description:
          "Average atmospheric pressure on the day, in inches of mercury.",
        dataType: "Continuous",
        example: "28.96 inHg",
      },
      {
        name: "Press_min",
        description:
          "Minimum atmospheric pressure on the day, in inches of mercury.",
        dataType: "Continuous",
        example: "29.63 inHg",
      },
      {
        name: "Precipit",
        description: "Total precipitation on the day, in inches.",
        dataType: "Continuous",
        example: "1.01 inches",
      },
    ],
  };

  return (
    <Container fluid>
      <h1 className="my-4">Handling Duplicate Entries</h1>

      <Row>
        <Col>
          <h2 id="types-of-duplicates">Types of Duplicates</h2>
          <p>
            Understanding the nature of duplicates is crucial for effective data
            cleaning:
          </p>
          <ul>
            <li>
              <strong>Exact Duplicates:</strong> Records that are identical
              across all features. Often arise from data entry errors or data
              merging processes.
            </li>
            <li>
              <strong>Partial Duplicates:</strong> Records that are identical in
              key fields but differ in others. They may occur due to
              inconsistent data collection or merging of similar datasets.
            </li>
            <li>
              <strong>Approximate Duplicates:</strong> Records that are not
              identical but very similar, often due to typos or different data
              entry standards.
            </li>
          </ul>

          <h2 id="identifying-duplicates">Identifying Duplicates</h2>
          <p>
            The first step in handling duplicates is identifying them through
            various methods depending on their nature.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd

# For exact duplicates
exact_duplicates = df[df.duplicated()]

# For partial duplicates, specify columns
partial_duplicates = df[df.duplicated(subset=['column1', 'column2'])]

print("Exact duplicates:", exact_duplicates.shape[0])
print("Partial duplicates:", partial_duplicates.shape[0])`}
          />
          <h2 id="visualize-duplicates">Visualize Duplicates</h2>
          <p>
            Visualizing duplicates can provide insightful perspectives on the
            distribution and impact of duplicate data within your dataset. This
            visualization helps in identifying patterns that might influence the
            handling strategy for duplicates, especially when deciding whether
            to remove or modify them.
          </p>
          <CodeBlock
            language={"python"}
            code={`import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
# Creating a temporary column 'is_duplicate' to mark duplicate rows
df['is_duplicate'] = df.duplicated(keep=False)

# Plotting duplicates
plt.figure(figsize=(10, 6))
sns.countplot(x='is_duplicate', data=df)
plt.title('Visualization of Duplicate Records')
plt.xlabel('Is Duplicate')
plt.ylabel('Count')
plt.show()

# Dropping the temporary column after visualization
df.drop(columns=['is_duplicate'], inplace=True)`}
          />

          <p>
            This visualization uses a simple count plot to show the presence of
            duplicate entries in the dataset. It marks each row as a duplicate
            or not and counts the occurrences, providing a clear visual
            representation of how many entries are affected. This method is
            particularly useful for quickly assessing the extent of duplication
            and determining if further cleaning steps are necessary.
          </p>

          <h2 id="removing-duplicates">Removing Duplicates</h2>
          <p>
            Removing duplicates should be tailored based on the type identified
            and the specific needs of your dataset:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Removing exact duplicates
df = df.drop_duplicates()

# Keeping the last occurrence of partial duplicates
df = df.drop_duplicates(subset=['column1', 'column2'], keep='last')`}
          />

          <h2 id="advanced-techniques">Advanced Techniques</h2>
          <p>
            For more complex scenarios, such as approximate duplicates, advanced
            techniques like fuzzy matching might be required:
          </p>
          <CodeBlock
            language={"python"}
            code={`from fuzzywuzzy import process

# Example of using fuzzy matching to find close matches
choices = df['column_name'].unique()
matches = process.extract('search_term', choices, limit=10)
print(matches)`}
          />

          <h2 id="considerations">Considerations</h2>
          <p>
            Consider the implications of removing duplicates in your data
            analysis. It’s essential to understand why duplicates appear and
            confirm that their removal is justified:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Considerations for time-series data
if 'date' in df.columns:
    df = df.drop_duplicates(subset=['date', 'category'], keep='first')`}
          />
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

export default HandleDuplicates;
