import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Introduction to Cloud Computing and GCP</h1>
      <p>
        In this section, you will learn about the basics of cloud computing and
        the specific offerings of Google Cloud Platform (GCP).
      </p>
      <Row>
        <Col>
          <h2>
            Overview of Cloud Computing: Benefits and Models (IaaS, PaaS, SaaS)
          </h2>
          <p>
            Cloud computing is a model for delivering on-demand computing
            resources over the internet. It offers benefits such as scalability,
            flexibility, and cost-effectiveness. There are three main service
            models: Infrastructure as a Service (IaaS), Platform as a Service
            (PaaS), and Software as a Service (SaaS).
          </p>
          <h2>
            Introduction to GCP and its Core Components (Compute Engine, App
            Engine, Kubernetes Engine)
          </h2>
          <p>
            Google Cloud Platform (GCP) is a cloud computing service offered by
            Google. Its core components include Compute Engine for virtual
            machines, App Engine for building and deploying web applications,
            and Kubernetes Engine for managing containerized applications.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
