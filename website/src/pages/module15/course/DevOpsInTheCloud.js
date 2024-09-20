import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DevOpsInTheCloud = () => {
  return (
    <Container fluid>
      <h1 className="my-4">DevOps in the Cloud</h1>
      <p>
        In this section, you will learn how to implement DevOps practices using
        GCP tools.
      </p>
      <Row>
        <Col>
          <h2>
            Introduction to Cloud Build, Container Registry, and Kubernetes
            Engine
          </h2>
          <p>
            Cloud Build is a service that allows you to build and deploy
            containers. Container Registry is a service that stores and manages
            container images. Kubernetes Engine is a service that allows you to
            run containers on a managed cluster.
          </p>
          <h2>Setting up CI/CD Pipelines with Cloud Build</h2>
          <p>
            CI/CD pipelines can be set up using Cloud Build to automate the
            build, test, and deployment of applications.
          </p>
          <CodeBlock
            code={`# Example of a Cloud Build configuration file
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/my-project/my-image', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/my-project/my-image']`}
          />
          <h2>Monitoring and Logging with Stackdriver</h2>
          <p>
            Stackdriver is a service that allows you to monitor and log the
            performance and health of your applications.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default DevOpsInTheCloud;
