import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Collaborating = () => {
  return (
    <Container fluid>
      <h2>Collaborating with Git</h2>
      <p>
        Collaboration in software development is crucial for scaling projects
        and improving code quality. Git, alongside hosting services like GitHub,
        provides a powerful set of tools to enhance collaboration among
        developers.
      </p>

      {/* Git Workflows */}
      <Row>
        <Col>
          <h3 id="git-workflows">Git Workflows</h3>
          <p>
            Git workflows define a consistent way to handle code changes in
            projects. The choice of workflow can affect the productivity of a
            team and the quality of a codebase. Here are some common Git
            workflows used in the industry:
          </p>
          <ul>
            <li>
              <strong>Feature Branch Workflow:</strong> Developers create new
              branches for each new feature, ensuring the master branch always
              contains production-quality code.
            </li>
            <li>
              <strong>Gitflow Workflow:</strong> This is an extension of the
              feature branch workflow, adding designated branches for preparing,
              maintaining, and recording releases.
            </li>
          </ul>
        </Col>
      </Row>

      {/* Code Reviews and Peer Review */}
      <Row className="mt-4">
        <Col>
          <h3 id="code-reviews">Code Reviews and Peer Reviews</h3>
          <p>
            Code reviews are a critical part of any collaborative project. They
            involve scrutinizing code changes by one or more developers other
            than the author:
          </p>
          <ol>
            <li>
              <strong>Submit a Pull Request (PR):</strong> The developer pushes
              their branch to the remote repository and opens a PR against the
              main branch. This starts the review process.
            </li>
            <li>
              <strong>Review Process:</strong> Team members comment on the code,
              suggest improvements, and discuss potential issues.
            </li>
            <li>
              <strong>Revise and Push:</strong> Based on feedback, the developer
              makes the necessary revisions and updates the PR.
            </li>
            <li>
              <strong>Approval and Merge:</strong> Once approved, the PR is
              merged into the main branch, integrating the changes.
            </li>
          </ol>
          <p>
            Effective code reviews catch bugs, improve code quality, and spread
            knowledge across the team.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default Collaborating;
