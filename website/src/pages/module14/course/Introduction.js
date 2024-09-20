import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Introduction = () => {
  return (
    <Container>
      <h1>Introduction to Containerization</h1>

      <section id="what-is-containerization">
        <h2>What is Containerization?</h2>
        <p>
          Containerization is a lightweight form of virtualization that packages
          an application and its dependencies into a standalone, portable unit
          called a container. This container can run consistently across
          different computing environments.
        </p>
      </section>

      <section id="benefits-of-containerization">
        <h2>Benefits of Containerization</h2>
        <ul>
          <li>
            <strong>Consistency:</strong> Containers ensure that applications
            run the same regardless of where they're deployed.
          </li>
          <li>
            <strong>Efficiency:</strong> Containers share the host OS kernel,
            making them more lightweight than traditional VMs.
          </li>
          <li>
            <strong>Portability:</strong> Containers can be easily moved between
            different environments, from development to production.
          </li>
          <li>
            <strong>Scalability:</strong> Containerized applications can be
            scaled up or down quickly to meet demand.
          </li>
          <li>
            <strong>Isolation:</strong> Containers provide a level of isolation
            between applications, improving security and resource management.
          </li>
        </ul>
      </section>

      <section id="containers-vs-vms">
        <h2>Containers vs. Virtual Machines</h2>
        <p>
          While both containers and virtual machines (VMs) are used for
          virtualization, they have key differences:
        </p>
        <Row>
          <Col md={6}>
            <h3>Containers</h3>
            <ul>
              <li>Share the host OS kernel</li>
              <li>Lightweight (MBs in size)</li>
              <li>Start up in seconds</li>
              <li>Less resource-intensive</li>
            </ul>
          </Col>
          <Col md={6}>
            <h3>Virtual Machines</h3>
            <ul>
              <li>Run a full copy of an OS</li>
              <li>Heavier (GBs in size)</li>
              <li>Take minutes to start</li>
              <li>More resource-intensive</li>
            </ul>
          </Col>
        </Row>
      </section>

      <section id="docker-overview">
        <h2>Docker Overview</h2>
        <p>
          Docker is a popular platform for developing, shipping, and running
          applications in containers. It provides tools and services to manage
          containers easily.
        </p>
        <h3>Key Docker Components:</h3>
        <ul>
          <li>
            <strong>Docker Engine:</strong> The runtime that runs and manages
            containers.
          </li>
          <li>
            <strong>Docker Images:</strong> Read-only templates used to create
            containers.
          </li>
          <li>
            <strong>Dockerfiles:</strong> Text files that contain instructions
            for building Docker images.
          </li>
          <li>
            <strong>Docker Hub:</strong> A cloud-based registry for sharing and
            storing Docker images.
          </li>
        </ul>
        <h3>Basic Docker Workflow:</h3>
        <CodeBlock
          code={`
# Build an image from a Dockerfile
docker build -t my-app .

# Run a container from the image
docker run -d -p 8080:80 my-app

# Stop the container
docker stop <container_id>

# Remove the container
docker rm <container_id>
          `}
          language="bash"
        />
      </section>

      <section id="conclusion">
        <h2>Conclusion</h2>
        <p>
          Containerization, and Docker in particular, has revolutionized how we
          develop, deploy, and scale applications. As we progress through this
          course, we'll dive deeper into Docker's features and best practices
          for using containers in various scenarios.
        </p>
      </section>
    </Container>
  );
};

export default Introduction;
