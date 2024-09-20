import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const NetworkingSecurity = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Networking and Security</h1>
      <p>
        In this section, you will learn about the networking capabilities and
        security protocols within GCP.
      </p>
      <Row>
        <Col>
          <h2>Configuring Virtual Private Cloud (VPC) and Firewalls</h2>
          <p>
            Virtual Private Cloud (VPC) is a service that allows you to create a
            private network for your resources in GCP. Firewalls are used to
            control the traffic that is allowed to enter and leave the network.
          </p>
          <CodeBlock
            code={`# Example of creating a VPC and firewall rules
gcloud compute networks create my-network
gcloud compute firewall-rules create my-firewall-rule --network my-network --allow tcp:80`}
          />
          <h2>Identity and Access Management (IAM): Roles and Permissions</h2>
          <p>
            IAM is used to manage access to resources in GCP. Roles define a set
            of permissions that can be granted to users, groups, or service
            accounts.
          </p>
          <CodeBlock
            code={`# Example of granting a role to a user
gcloud projects add-iam-policy-binding my-project --member="user:john@example.com" --role="roles/editor"`}
          />
          <h2>Best Practices for Securing Applications on GCP</h2>
          <p>
            Securing applications on GCP involves using VPCs and firewalls to
            control network traffic, using IAM to manage access to resources,
            and using encryption to protect data at rest and in transit.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default NetworkingSecurity;
