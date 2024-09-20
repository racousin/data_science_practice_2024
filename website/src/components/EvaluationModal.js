import React, { useState } from "react";
import { Button, Modal, Container, Row, Col, Alert } from "react-bootstrap";
import CodeBlock from "components/CodeBlock"; // Ensure you have a component to render code blocks

const EvaluationModal = ({ module }) => {
  const [show, setShow] = useState(false);

  const handleOpen = () => setShow(true); // Open modal on click
  const handleClose = () => setShow(false); // Close modal

  return (
    <>
      <Button
        variant="outline-secondary"
        onClick={handleOpen}
        className="nav-button button-outline"
        aria-expanded={show}
        aria-controls="evaluation-modal"
      >
        Submit your exercises
      </Button>

      <Modal
        show={show}
        onHide={handleClose}
        centered
        id="evaluation-modal"
        size="lg"
        aria-labelledby="modal-title"
      >
        <Modal.Header closeButton>
          <Modal.Title id="modal-title">
            Exercise Submission Guidelines
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <Container>
            <Row>
              <Col>
                <h2 className="my-3">Prerequisites</h2>
                <Alert variant="primary">
                  Make sure to complete the initial setup exercises in{" "}
                  <a
                    href="/module0/course"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Module 0
                  </a>{" "}
                  before proceeding.
                </Alert>
              </Col>
            </Row>
            <h2 className="my-4">Steps to Submit Your Exercise</h2>
            <Row>
              <Col>
                <h3>Initial Setup</h3>
                <p>Navigate to your project directory:</p>
                <CodeBlock code={`cd your_project_directory`} />
                <p>
                  Ensure you are on the main branch and your repository is up to
                  date:
                </p>
                <CodeBlock
                  code={`git checkout main
git pull origin main`}
                />
              </Col>
            </Row>
            <Row>
              <Col>
                <h3>Prepare Your Work</h3>
                <p>Create and switch to a new branch for your exercise:</p>
                <CodeBlock
                  code={`git checkout -b exercise_branch/$username/module${module}`}
                />
                <p>
                  Create a directory for your module (if it doesn't already
                  exist):
                </p>
                <CodeBlock code={`mkdir -p $username/module${module}`} />
                <p>Perform your work in this directory.</p>
              </Col>
            </Row>
            <Row>
              <Col>
                <h3>Submit Your Exercise</h3>
                <p>Stage your changes for commit:</p>
                <CodeBlock
                  code={`git add $username/module${module}/your_files`}
                />
                <p>Commit your changes:</p>
                <CodeBlock code={`git commit -m 'Update exercise files'`} />
                <p>Push your branch to the repository:</p>
                <CodeBlock
                  code={`git push origin exercise_branch/$username/module${module}`}
                />
              </Col>
            </Row>
            <Row>
              <Col>
                <h3>Create a Pull Request</h3>
                <p>After pushing your work to your remote branch:</p>
                <ul>
                  <li>
                    Visit{" "}
                    <a
                      href="https://github.com/racousin/data_science_practice/pulls"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      GitHub Pull Requests
                    </a>
                    .
                  </li>
                  <li>
                    Create a new pull request from your exercise branch to the
                    main branch.
                  </li>
                  <li>
                    Request a review and make necessary changes based on
                    feedback.
                  </li>
                  <li>Merge the pull request once approved.</li>
                </ul>
                <p>
                  After merging, you can consult your results in the{" "}
                  <a
                    href="/repositories"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    repository results page
                  </a>
                  .
                </p>
              </Col>
            </Row>
          </Container>
        </Modal.Body>
      </Modal>
    </>
  );
};

export default EvaluationModal;
