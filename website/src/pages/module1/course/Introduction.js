import React from "react";
import { Container, Row, Col, Image } from "react-bootstrap";

const Introduction = () => {
  return (
    <Container fluid>
      <h2>Introduction</h2>
      <Row>
        <Col md={12}>
          <p>
            Git is a distributed version control system widely used to
            coordinate work among programmers. It tracks changes in source code
            during software development, allowing for efficient collaboration
            and historical referencing.
          </p>
          <p>
            Developed by Linus Torvalds in 2005 for Linux kernel development,
            Git has since become essential for managing projects ranging from
            small teams to large enterprises. It supports non-linear development
            through its robust branching and merging capabilities, enabling
            multiple parallel workflows.
          </p>
          <h4>Why Use Version Control?</h4>
          <p>
            Version control systems are fundamental in software development for
            maintaining a clear history of code changes, facilitating
            collaborative adjustments, and ensuring that earlier versions of
            work can be retrieved. This is crucial in complex projects where
            tracking the contributions of each team member is necessary for
            effective progression.
          </p>
          <h4>Git Platforms in Modern Development</h4>
          <p>
            Git's impact extends beyond just tracking changes. It integrates
            with various services to enhance project management capabilities.
            Here are a few key platforms:
          </p>
          <ul>
            <li>
              <a href="https://github.com">GitHub</a>
            </li>
            <li>
              <a href="https://gitlab.com">GitLab</a>
            </li>
            <li>
              <a href="https://bitbucket.org">Bitbucket</a>
            </li>
            <li>
              <a href="https://aws.amazon.com/codecommit/">AWS CodeCommit</a>
            </li>
            <li>
              <a href="https://cloud.google.com/source-repositories">
                Google Cloud Source Repositories
              </a>
            </li>
            <li>
              <a href="https://azure.microsoft.com/en-us/services/devops/repos/">
                Azure Repos
              </a>
            </li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default Introduction;
