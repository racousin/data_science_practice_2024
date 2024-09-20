import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DataQuality = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Data Quality and Validation</h1>

      <section>
        <h2 id="assessing-quality">Assessing Data Quality</h2>
        <p>
          Assessing data quality is crucial for ensuring the reliability and
          validity of your analysis. Here are some key aspects to consider:
        </p>
        <ul>
          <li>
            <strong>Completeness:</strong> Are there missing values?
          </li>
          <li>
            <strong>Accuracy:</strong> Is the data correct and reliable?
          </li>
          <li>
            <strong>Consistency:</strong> Is the data consistent across the
            dataset?
          </li>
          <li>
            <strong>Timeliness:</strong> Is the data up-to-date?
          </li>
          <li>
            <strong>Uniqueness:</strong> Are there duplicate records?
          </li>
          <li>
            <strong>Validity:</strong> Does the data conform to the required
            format or range?
          </li>
        </ul>
      </section>

      <section>
        <h2 id="profiling">Data Profiling Techniques</h2>
        <p>
          Data profiling involves analyzing the content, structure, and
          relationships within your data. It can help identify data quality
          issues and provide insights into your dataset.
        </p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
import pandas_profiling

def profile_data(df):
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file("data_profile_report.html")
    return profile

# Usage
df = pd.read_csv('your_data.csv')
profile = profile_data(df)
profile.to_notebook_iframe()  # Display in Jupyter Notebook
          `}
        />
        <p>
          The pandas-profiling library generates a comprehensive report
          including:
        </p>
        <ul>
          <li>
            Dataset overview (number of variables, observations, missing cells,
            duplicates)
          </li>
          <li>Variable types and statistics</li>
          <li>Correlations between variables</li>
          <li>Missing value patterns</li>
          <li>Distribution of variables</li>
        </ul>
      </section>

      <section>
        <h2 id="validation-checks">Implementing Data Validation Checks</h2>
        <p>
          Data validation involves checking if your data adheres to a set of
          rules or constraints. Here's how you can implement some common
          validation checks:
        </p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
from pandas.api.types import is_numeric_dtype

def validate_data(df, rules):
    validation_results = {}
    
    for column, rule in rules.items():
        if column not in df.columns:
            validation_results[column] = "Column not found"
            continue
        
        if 'type' in rule:
            if rule['type'] == 'numeric' and not is_numeric_dtype(df[column]):
                validation_results[column] = "Not numeric type"
        
        if 'range' in rule and is_numeric_dtype(df[column]):
            min_val, max_val = rule['range']
            if df[column].min() < min_val or df[column].max() > max_val:
                validation_results[column] = f"Out of range {min_val}-{max_val}"
        
        if 'unique' in rule and rule['unique']:
            if df[column].duplicated().any():
                validation_results[column] = "Contains duplicates"
        
        if 'regex' in rule:
            if not df[column].astype(str).str.match(rule['regex']).all():
                validation_results[column] = "Does not match regex pattern"
    
    return validation_results

# Usage
df = pd.read_csv('your_data.csv')
rules = {
    'age': {'type': 'numeric', 'range': (0, 120)},
    'email': {'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
    'id': {'unique': True}
}
validation_results = validate_data(df, rules)
print(validation_results)
          `}
        />
      </section>

      <section>
        <h2>Advanced Data Quality Techniques</h2>
        <h3>Anomaly Detection</h3>
        <p>
          Anomaly detection can help identify unusual patterns that don't
          conform to expected behavior.
        </p>
        <CodeBlock
          language="python"
          code={`
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, columns):
    clf = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = clf.fit_predict(df[columns])
    return df[df['anomaly'] == -1]

# Usage
anomalies = detect_anomalies(df, ['numeric_column1', 'numeric_column2'])
print(anomalies)
          `}
        />

        <h3>Data Quality Scoring</h3>
        <p>
          Assigning a quality score to your dataset can provide a quick overview
          of its overall quality.
        </p>
        <CodeBlock
          language="python"
          code={`
def calculate_quality_score(df):
    completeness = 1 - df.isnull().mean().mean()
    uniqueness = 1 - df.duplicated().mean()
    
    # Simple validity check (assuming all columns should be non-negative)
    validity = (df >= 0).mean().mean()
    
    quality_score = (completeness + uniqueness + validity) / 3
    return quality_score

# Usage
quality_score = calculate_quality_score(df)
print(f"Data Quality Score: {quality_score:.2f}")
          `}
        />
      </section>

      <section>
        <h2>Best Practices for Ensuring Data Quality</h2>
        <ul>
          <li>
            <strong>Regular Audits:</strong> Conduct regular data quality audits
            to catch issues early.
          </li>
          <li>
            <strong>Data Governance:</strong> Implement data governance policies
            to maintain data quality over time.
          </li>
          <li>
            <strong>Automated Checks:</strong> Set up automated data quality
            checks in your data pipeline.
          </li>
          <li>
            <strong>Documentation:</strong> Maintain clear documentation of data
            sources, transformations, and quality checks.
          </li>
          <li>
            <strong>Feedback Loop:</strong> Establish a feedback loop with data
            users to report and address quality issues.
          </li>
          <li>
            <strong>Data Lineage:</strong> Track data lineage to understand how
            data flows and transforms through your systems.
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default DataQuality;
