import React from "react";
import { Container, Row, Col, Table } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleCategoricalValues = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handling_categorical.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handling_categorical.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handling_categorical.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/handling_categorical.ipynb";
  const metadata = {
    description:
      "This dataset includes characteristics of different mushrooms, aiming to classify them as either poisonous or edible based on various physical attributes.",
    source: "Mycology Research Data",
    target: "class",
    listData: [
      {
        name: "class",
        description:
          "Indicates whether the mushroom is poisonous (p) or edible (e).",
        dataType: "Categorical",
        example: "p (poisonous), e (edible)",
      },
      {
        name: "cap-diameter",
        description:
          "Numerical measurement likely representing the diameter of the mushroom's cap.",
        dataType: "Continuous",
        example: "15 cm",
      },
      {
        name: "cap-shape",
        description:
          "Descriptive categories for the shape of the mushroom cap (e.g., x for convex, f for flat).",
        dataType: "Categorical",
        example: "x (convex)",
      },
      {
        name: "stem-width",
        description:
          "Numerical measurement likely representing the width of the mushroom's stem.",
        dataType: "Continuous",
        example: "2 cm",
      },
      {
        name: "has-ring",
        description:
          "Boolean indicating the presence of a ring (t for true, f for false).",
        dataType: "Boolean",
        example: "t (true)",
      },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">Handling Categorical Values</h1>

      <Row>
        <Col>
          <h2 id="types-of-categorical-data">Types of Categorical Data</h2>
          <p>
            There are two main types of categorical data:{" "}
            <strong>nominal</strong>, where data categories do not have an
            inherent order (e.g., colors, types of cuisine), and{" "}
            <strong>ordinal</strong>, where the categories have a logical order
            (e.g., rankings, education level).
          </p>
          <h2 id="identify-and-visualize-categorical-values">
            Identify and Visualize Categorical Values
          </h2>
          <p>
            Identifying and visualizing categorical data is a crucial step in
            data preprocessing, helping to understand the distribution of
            categories and their influence on the dataset. Visualizing
            categorical data can highlight imbalances or patterns that might
            affect model performance. Enhanced visualization techniques like
            count plots and bar plots can provide deeper insights by comparing
            category distributions across different subgroups or in relation to
            a target variable.
          </p>
          <CodeBlock
            language={"python"}
            code={`import seaborn as sns
import matplotlib.pyplot as plt

# Identifying categorical columns by checking unique values or data type
categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() < 10]

# Visualizing the distribution of a categorical column with a count plot
plt.figure(figsize=(10, 6))
sns.countplot(x='category_column', data=df, palette='Blues')
plt.title('Count Plot of Categorical Data')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotates labels to avoid overlap
plt.show()

# Using a bar plot to visualize the relationship between a categorical variable and a target variable
plt.figure(figsize=(12, 8))
sns.barplot(x='category_column', y='target_column', data=df, ci=None, palette='viridis')
plt.title('Bar Plot Showing Relationship Between Categories and Target')
plt.xlabel('Categories')
plt.ylabel('Average of Target Variable')
plt.xticks(rotation=45)
plt.show()

# Advanced visualization with count plot showing distribution across an additional category
fig = plt.figure(figsize=(12,8))
sns.countplot(x='category_column', hue='target_column', data=df, palette='Blues');
fig.autofmt_xdate()
plt.title('Distribution of Categories by Target')
plt.xlabel('Categories')
plt.ylabel('Frequency of target_column')
plt.legend(title='Transmission')
plt.show()`}
          />

          <p>
            These visualizations offer a comprehensive view of categorical data
            distribution, both independently and in relation to other variables.
            Count plots provide a straightforward visual summary of frequency
            distribution across categories, while bar plots highlight the impact
            of categorical variables on a continuous or ordinal target,
            facilitating deeper insights into potential relationships within the
            data.
          </p>

          <h2 id="one-hot-encoding">One-Hot Encoding</h2>
          <p>
            One of the most common methods for handling nominal categorical
            data. It involves creating a new binary column for each category and
            is particularly useful for models that expect numerical input, like
            logistic regression and neural networks.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd

df = pd.get_dummies(df, columns=['your_column'])`}
          />

          <h2 id="label-encoding">Label Encoding</h2>
          <p>
            Converts each value in a column to a number. Useful for ordinal data
            as the numbers can represent the order of the categories.
          </p>
          <CodeBlock
            language={"python"}
            code={`from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['your_column'] = le.fit_transform(df['your_column'])`}
          />

          <h2 id="handling-unseen-categories">Handling Unseen Categories</h2>
          <p>
            When your model encounters a category in the test set that wasn't
            present during training, it's important to have a strategy to handle
            this. Common approaches include assigning an 'unknown' label or
            using the most frequent category.
          </p>

          <h2 id="feature-engineering">
            Feature Engineering with Categorical Data
          </h2>
          <p>
            Combining categories, creating interaction terms, or extracting
            information from mixed data types (like addresses) can create
            valuable features that improve model performance.
          </p>

          <h2 id="use-in-models">Using Categorical Data in Models</h2>
          <p>
            Many machine learning algorithms can now directly handle categorical
            data. Libraries like CatBoost or handling in algorithms like
            decision trees do not require explicit encoding as these algorithms
            can inherently deal with categorical variables.
          </p>
        </Col>
      </Row>
      <Row>
        <div id="notebook-example"></div>
        <DataInteractionPanel
          DataUrl={DataUrl}
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

export default HandleCategoricalValues;
