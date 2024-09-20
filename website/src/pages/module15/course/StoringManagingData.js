import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const StoringManagingData = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Storing and Managing Data</h1>
      <p>In this section, you will explore data storage options within GCP.</p>
      <Row>
        <Col>
          <h2>Introduction to Cloud Storage, Datastore, and Bigtable</h2>
          <p>
            Cloud Storage is a service for storing and retrieving objects, such
            as images, videos, and documents. Datastore is a fully managed,
            scalable database for storing structured data. Bigtable is a
            scalable, high-performance database for storing large amounts of
            data.
          </p>
          <h2>Best Practices for Data Security and Management</h2>
          <p>
            Best practices for data security include using encryption,
            implementing access controls, and monitoring access logs. Best
            practices for data management include organizing data into logical
            groups, using consistent naming conventions, and versioning data.
          </p>
          <h2>
            Integrating Cloud SQL and Cloud Spanner for Database Management
          </h2>
          <p>
            Cloud SQL is a fully managed database service that supports MySQL,
            PostgreSQL, and SQL Server. Cloud Spanner is a globally distributed,
            multi-version, synchronously replicated database.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default StoringManagingData;
