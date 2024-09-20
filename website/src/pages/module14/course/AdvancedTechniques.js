import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const AdvancedTechniques = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Advanced Docker Techniques</h1>
      <p>
        In this section, you will learn about more complex Docker
        functionalities.
      </p>
      <Row>
        <Col>
          <h2>Building Optimized Docker Images for Applications</h2>
          <p>
            Building optimized Docker images for applications can improve
            performance and security. This can be done by using multi-stage
            builds, minimizing the size of the image, and using secure base
            images.
          </p>
          <CodeBlock
            code={`# Example of a multi-stage build
FROM python:3.8-slim-buster as builder
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN pip install .

FROM python:3.8-slim-buster
WORKDIR /app
COPY --from=builder /app/ /app/`}
          />
          <h2>Docker Networking in Depth</h2>
          <p>
            Docker provides several networking options for containers. These
            include bridge networks, host networks, and overlay networks.
            Understanding these options can help you to configure networking for
            your containers to meet your needs.
          </p>
          <h2>Using Docker with Cloud Services (e.g., AWS, Azure)</h2>
          <p>
            Docker can be used with cloud services to deploy and manage
            containers. This can be done by using a cloud provider's container
            service, such as Amazon Elastic Container Service (ECS) or Azure
            Kubernetes Service (AKS).
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default AdvancedTechniques;
