import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const HandleInconsistencies = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_handle_inconsistencies.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handle_inconsistencies.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/handle_inconsistencies.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/handle_inconsistencies.ipynb";
  const metadata = {
    description:
      "This dataset contains demographic information including state, country, age, and date.",
    source: "Demographic Survey Records",
    target: null,
    listData: [
      {
        name: "State",
        description: "The state or province of residence",
      },
      {
        name: "Country",
        description: "The country of residence",
      },
      {
        name: "Age",
        description: "The age of the individual in years",
      },
      {
        name: "Date",
        description: "The date the information was recorded (YYYY-MM-DD)",
      },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">Handle Inconsistencies</h1>

      <Row>
        <Col>
          <h2 id="types-of-inconsistencies">Types of Inconsistencies</h2>
          <p>Data inconsistencies can manifest in several ways:</p>
          <ul>
            <li>
              <strong>Formatting inconsistencies:</strong> Variations in date
              formats, text capitalization, or use of special characters.
            </li>
            <li>
              <strong>Type inconsistencies:</strong> Mixed data types within a
              column (e.g., numbers and strings).
            </li>
            <li>
              <strong>Duplicate records:</strong> Repeated entries that may or
              may not be exact duplicates.
            </li>
            <li>
              <strong>Contradictory data:</strong> Records that conflict with
              each other, often due to data entry errors or merging issues.
            </li>
          </ul>

          <h2 id="detecting-inconsistencies">Detecting Inconsistencies</h2>
          <p>Effective detection is the first step towards cleaning:</p>
          <CodeBlock
            language={"python"}
            code={`# Detecting type inconsistencies
df['column'].apply(type).value_counts()

# Detecting formatting issues
df['date'].apply(lambda x: isinstance(x, str) and not re.match(r'\\d{4}-\\d{2}-\\d{2}', x)).sum()`}
          />

          <h2 id="solutions-to-inconsistencies">
            Solutions to Inconsistencies
          </h2>
          <p>
            Depending on the issue, you may choose to treat, remove, or modify
            inconsistent data:
          </p>
          <ul>
            <li>
              <strong>Casting types:</strong> Ensure all data in a column is of
              the same type.
            </li>
            <li>
              <strong>Standardizing text:</strong> Convert all text data to a
              consistent format (e.g., all lower case).
            </li>
            <li>
              <strong>Removing or correcting outliers:</strong> Based on
              business rules or statistical methods.
            </li>
          </ul>
          <CodeBlock
            language={"python"}
            code={`# Casting types
df['numeric_column'] = pd.to_numeric(df['numeric_column'], errors='coerce')

# Standardizing text
df['text_column'] = df['text_column'].str.lower().str.strip().replace(r'\\s+', ' ', regex=True)`}
          />

          <h2 id="advanced-text-manipulation">Advanced Text Manipulation</h2>
          <p>
            Manage special characters and whitespace to improve text data
            quality:
          </p>
          <CodeBlock
            language={"python"}
            code={`# Remove special characters
df['text_column'] = df['text_column'].str.replace(r'[^\\w\\s]', '', regex=True)`}
          />
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

export default HandleInconsistencies;
