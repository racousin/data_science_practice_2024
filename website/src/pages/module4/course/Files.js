import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Files = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Files</h1>
      <p>
        In this section, you will learn how to retrieve data from files using
        Python.
      </p>
      <Row>
        <Col>
          <h2>CSV Files</h2>
          <p>
            CSV (Comma Separated Values) files are a common format for storing
            data in a tabular format. To read a CSV file in Python, you can use
            the `pandas` library's `read_csv()` function.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd

data = pd.read_csv("data.csv")`}
          />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Excel Files</h2>
          <p>
            Excel files are a common format for storing data in a spreadsheet
            format. To read an Excel file in Python, you can use the `pandas`
            library's `read_excel()` function.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd

data = pd.read_excel("data.xlsx")`}
          />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>Parquet Files</h2>
          <p>
            Parquet files are a columnar storage format that is optimized for
            use with big data processing frameworks like Apache Hadoop and
            Apache Spark. To read a Parquet file in Python, you can use the
            `pandas` library's `read_parquet()` function.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd

data = pd.read_parquet("data.parquet")`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default Files;
