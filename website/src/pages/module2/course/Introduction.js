import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h2>Introduction</h2>
      <Row>
        <Col md={12}>
          <h4>Why Use Python Environments?</h4>
          <p>
            Python environments are isolated contexts where Python packages and
            dependencies are installed. This isolation prevents version
            conflicts and ensures that projects can be developed and tested in
            settings that closely mimic their production environments.
          </p>
          <h4>The Importance of Package Management</h4>
          <p>
            Package management involves organizing, installing, and maintaining
            software libraries that projects depend on. Python’s package
            ecosystem includes thousands of third-party modules available on the
            Python Package Index (PyPI), which can be managed using tools like
            pip and conda.
          </p>
          <h4>Key Tools for Python Environments and Package Management</h4>
          <p>
            The following tools are commonly used for managing Python
            environments and packages:
          </p>
          <ul>
            <li>
              <strong>pip:</strong> Python’s standard package-management system
              used to install and manage software packages.
            </li>
            <li>
              <strong>virtualenv:</strong> A tool to create isolated Python
              environments.
            </li>
            <li>
              <strong>conda:</strong> An open-source package management system
              and environment management system.
            </li>
            <li>
              <strong>Poetry:</strong> An open-source package management system
              and environment management system.
            </li>
            <li>
              <strong>Pipenv:</strong> An open-source package management system
              and environment management system.
            </li>
          </ul>
        </Col>
      </Row>
      {/* <Row>
        <Col md={10}>
          <Image
            src="/assets/python-environments.png"
            alt="Python Environments and Package Management"
            fluid
          />
          <p className="text-center">
            Fig.1 - Python Environments and Package Management
          </p>
        </Col>
      </Row> */}
    </Container>
  );
};

export default Introduction;
