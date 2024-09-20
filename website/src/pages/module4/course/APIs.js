import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const APIs = () => {
  return (
    <Container fluid>
      <h1 className="my-4">APIs</h1>
      <p>
        In this section, you will learn how to retrieve data from APIs using
        Python.
      </p>
      <Row>
        <Col>
          <h2>REST APIs</h2>
          <p>
            REST (Representational State Transfer) APIs are a common format for
            exposing data over the web. To retrieve data from a REST API in
            Python, you can use the `requests` library.
          </p>
          <CodeBlock
            language={"python"}
            code={`import requests

response = requests.get("https://api.example.com/data")
data = response.json()`}
          />
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h2>GraphQL APIs</h2>
          <p>
            GraphQL APIs are a query language for APIs that allows clients to
            request exactly the data they need. To retrieve data from a GraphQL
            API in Python, you can use the `gql` and `requests` libraries.
          </p>
          <CodeBlock
            language={"python"}
            code={`from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

transport = RequestsHTTPTransport(url="https://api.example.com/graphql")
client = Client(transport=transport, fetch_schema_from_transport=True)

query = gql("""
  {
    data {
      id
      name
    }
  }
""")

data = client.execute(query)`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default APIs;
