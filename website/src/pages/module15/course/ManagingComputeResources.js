import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const ManagingComputeResources = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Managing Compute Resources</h1>
      <p>
        In this section, you will learn how to manage and scale compute
        resources using GCP.
      </p>
      <Row>
        <Col>
          <h2>Setting up and Managing Compute Engine Instances</h2>
          <p>
            Compute Engine is a service that provides virtual machines on
            Google's infrastructure. You can create and manage instances using
            the GCP Console or the gcloud command-line tool.
          </p>
          <CodeBlock
            code={`# Example of creating a Compute Engine instance using the gcloud command-line tool
gcloud compute instances create my-instance --machine-type=n1-standard-1 --image=debian-9-stretch-v20200902`}
          />
          <h2>Auto-scaling and Load Balancing</h2>
          <p>
            Auto-scaling allows you to automatically adjust the number of
            instances based on traffic. Load balancing distributes traffic
            across multiple instances to ensure high availability and
            performance.
          </p>
          <h2>Leveraging Preemptible VMs for Cost-effective Computing</h2>
          <p>
            Preemptible VMs are virtual machines that are available at a
            discounted price. They can be terminated by Google at any time, so
            they are suitable for workloads that can tolerate interruptions.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default ManagingComputeResources;
