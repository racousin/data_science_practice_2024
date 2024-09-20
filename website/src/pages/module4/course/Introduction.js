import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Data Collection</h1>

      <section>
        <h2 id="importance">
          Importance of Data Collection in the Data Science Pipeline
        </h2>
        <p>
          Data collection is a crucial first step in the data science pipeline.
          It forms the foundation upon which all subsequent analyses, models,
          and insights are built. The quality and relevance of the data
          collected directly impact the effectiveness and accuracy of data
          science projects.
        </p>
        <ul>
          <li>Enables data-driven decision making</li>
          <li>
            Provides the raw material for discovering patterns and insights
          </li>
          <li>
            Allows for the training and validation of machine learning models
          </li>
          <li>Facilitates the testing of hypotheses and theories</li>
        </ul>
      </section>

      <section>
        <h2 id="types">Types of Data</h2>
        <p>
          Data can be categorized into three main types based on its structure:
        </p>
        <h3>1. Structured Data</h3>
        <p>
          Organized in a predefined format, typically in tables with rows and
          columns.
        </p>
        <ul>
          <li>Examples: Relational databases, spreadsheets</li>
          <li>Easy to search and analyze</li>
          <li>Often numeric or categorical</li>
        </ul>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Example of structured data
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
}
df = pd.DataFrame(data)
print(df)
          `}
        />

        <h3>2. Semi-structured Data</h3>
        <p>
          Has some organizational properties but doesn't conform to a rigid
          structure.
        </p>
        <ul>
          <li>Examples: JSON, XML, email</li>
          <li>More flexible than structured data</li>
          <li>Can be parsed with some effort</li>
        </ul>
        <CodeBlock
          language="python"
          code={`
import json

# Example of semi-structured data (JSON)
json_data = '''
{
    "name": "Alice",
    "age": 25,
    "city": "New York",
    "hobbies": ["reading", "swimming", "photography"]
}
'''
parsed_data = json.loads(json_data)
print(parsed_data)
          `}
        />

        <h3>3. Unstructured Data</h3>
        <p>Data with no predefined format or organization.</p>
        <ul>
          <li>Examples: Text documents, images, audio files</li>
          <li>Most abundant form of data</li>
          <li>Requires advanced techniques for analysis</li>
        </ul>
        <CodeBlock
          language="python"
          code={`
# Example of handling unstructured data (text)
text_data = "This is an example of unstructured text data. It can contain any information without a specific format."

# Simple word count analysis
word_count = len(text_data.split())
print(f"Word count: {word_count}")
          `}
        />
      </section>

      <section>
        <h2 id="sources">Data Sources: Primary vs Secondary</h2>
        <p>
          Understanding the origin of data is crucial for assessing its
          reliability and applicability.
        </p>

        <h3>Primary Data Sources</h3>
        <p>Data collected firsthand for a specific purpose.</p>
        <ul>
          <li>Surveys and questionnaires</li>
          <li>Experiments and observations</li>
          <li>Interviews and focus groups</li>
          <li>Sensor data from IoT devices</li>
        </ul>
        <p>
          Advantages: Tailored to specific needs, high control over data quality
        </p>
        <p>Disadvantages: Can be time-consuming and expensive to collect</p>

        <h3>Secondary Data Sources</h3>
        <p>Pre-existing data collected for other purposes.</p>
        <ul>
          <li>Government databases</li>
          <li>Academic research data</li>
          <li>Commercial data providers</li>
          <li>Open data repositories</li>
        </ul>
        <p>Advantages: Often readily available, can save time and resources</p>
        <p>
          Disadvantages: May not perfectly fit the specific research needs,
          potential quality issues
        </p>

        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Example of accessing a secondary data source (CSV file from a public dataset)
url = "https://data.cityofnewyork.us/api/views/kku6-nxdu/rows.csv"
df = pd.read_csv(url)
print(df.head())
          `}
        />
      </section>
    </Container>
  );
};

export default Introduction;
