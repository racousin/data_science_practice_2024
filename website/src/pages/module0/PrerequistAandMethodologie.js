import React from "react";
import { Container, Row, Col, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import ModuleNavigation from "components/ModuleNavigation";
import CodeBlock from "components/CodeBlock";
import ModuleFrame from "components/ModuleFrame";

const PrerequisiteAndMethodology = () => {
  return (
    <ModuleFrame
      module={0}
      isCourse={null}
      title="Module 0: Prerequisites and Methodology"
      courseLinks={[]}
    >
      <Row>
        <Col>
          <p>Last Updated: {"2024-09-20"}</p>
        </Col>
      </Row>

      <Row>
        <Col md={11}>
          <h2>Setting Up Your Environment</h2>
          <p>
            To participate in the course exercises and submit your work, you
            need to set up a GitHub account and be added to the course's GitHub
            repository.
          </p>
          <ol>
            <li>
              <strong>Create a GitHub Account:</strong> If you do not have a
              GitHub account, you will need to create one. Follow the
              instructions on{" "}
              <a
                href="/module1/course/configure-and-access-github"
                target="_blank"
                rel="noopener noreferrer"
              >
                Creating a GitHub Account
              </a>
              .
            </li>
            <li>
              <strong>Email Registration for Course Repository:</strong> Once
              you have your GitHub username, send an email to{" "}
              <a href="mailto:raphaelcousin.education@gmail.com">
                raphaelcousin.education@gmail.com
              </a>{" "}
              with the subject
              "2024_github_account_registration:$name:$surname:$username",
              replacing $name, $surname, and $username with your actual details.
              You will receive an invitation to join the{" "}
              <a
                href="https://github.com/racousin/data_science_practice"
                target="_blank"
                rel="noopener noreferrer"
              >
                <i>data_science_practice</i>
              </a>{" "}
              repository in order to submit your exercises answers.
            </li>
            <li>
              After receiving the invitation, accept it to access and you can
              start with the <Link to="/module1/course/">Module 1: Git</Link>
            </li>
          </ol>
        </Col>
      </Row>

      <Row>
      <Col md={11}>
        <h2>Methodology and Submitting Exercises</h2>
        <p>
          Each module in this course is divided into a learning section and an
          exercise section.
        </p>
        <p>In each module, you will find:</p>
        <ul>
          <li>
            <strong>Mandatory exercises (e.g., Exercise1<span style={{color: 'red', fontWeight: 'bold'}}>*</span>):</strong> These exercises are required and will be evaluated.
          </li>
          <li>
            <strong>Optional exercises (e.g., Exercise2):</strong> You are encouraged to complete these for additional practice and learning.
          </li>
        </ul>
        <p>The objectives of submitting exercises are:</p>
        <ul>
          <li>
            <strong>Mastering Git and GitHub:</strong> Essential tools in the
            data science industry and in research.
          </li>
          <li>
            <strong>Understanding the Review Process:</strong> Critical for
            software development and fostering collaborative skills.
          </li>
          <li>
            <strong>Tracking Your Progress:</strong> After merging your
            changes to the main branch, CI/CD processes will validate your
            exercises. Successfully passing these tests will reflect on your
            student results page.
          </li>
        </ul>
        <p>
          To view your progress and results, visit the{" "}
          <a
            href="https://www.raphaelcousin.com/repositories"
            target="_blank"
            rel="noopener noreferrer"
          >
            Student Results Page
          </a>{" "}
          on the course website.
        </p>
      </Col>
    </Row>

      <Row>
        <h2>Example Exercise Submission Steps</h2>
        <ol>
          <li>
            <strong>If not done, clone the repository:</strong>
            <CodeBlock
              code={`git clone https://github.com/racousin/data_science_practice.git`}
            />
          </li>
          <li>
            <strong>
              If not done, configure Git with your GitHub account:
            </strong>
            <CodeBlock code={`git config --global user.name "Your Name"`} />
            <CodeBlock
              code={`git config --global user.email "your_email@example.com"`}
            />
          </li>
          <li>
            <strong>Pull the latest main branch:</strong>
            <CodeBlock code={`git checkout main`} />
            <CodeBlock code={`git pull origin main`} />
          </li>
          <li>
            <strong>Create and checkout to a new branch:</strong>
            <CodeBlock code={`git checkout -b myusername_branch/module1`} />
          </li>
          <li>
            <strong>Work on the exercise in your repository folder:</strong>
            <p>Example files and folder:</p>
            <ul>
              <li>myusername_branch/module1/file.py</li>
              <li>myusername_branch/module1/file.csv</li>
              <li>myusername_branch/module1/my_pkg/</li>
            </ul>
          </li>
          <li>
            <strong>Stage your changes:</strong>
            <CodeBlock code={`git add myusername_branch/module1`} />
          </li>
          <li>
            <strong>Commit your changes:</strong>
            <CodeBlock
              code={`git commit -m "My content for module 1 exercises"`}
            />
          </li>
          <li>
            <strong>Push your changes:</strong>
            <CodeBlock code={`git push -m origin myusername_branch/module1`} />
          </li>
          <li>
            <strong>Go to GitHub to create a new pull request:</strong>
            <p>
              Visit{" "}
              <a href="https://github.com/racousin/data_science_practice/pulls">
                GitHub Pull Requests
              </a>{" "}
              to create a new pull request from branch{" "}
              <code>myusername_branch/module1</code> to <code>main</code>.
            </p>
          </li>
          <li>
            <strong>Ask for reviewers:</strong>
            <p>Add designated reviewers to the pull request for feedback.</p>
          </li>
          <li>
            <strong>
              If needed, integrate the reviewers' changes and ask for a review
              again:
            </strong>
            <p>
              Update your branch with suggested changes and re-request a review.
            </p>
          </li>
          <li>
            <strong>Merge the pull request:</strong>
            <p>Once approved, merge your pull request into the main branch.</p>
          </li>
        </ol>
      </Row>
    </ModuleFrame>
  );
};

export default PrerequisiteAndMethodology;
