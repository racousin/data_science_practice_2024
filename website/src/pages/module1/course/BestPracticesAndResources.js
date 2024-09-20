import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BestPracticesAndResources = () => {
  return (
    <Container fluid>
      <h2>Best Practices and Resources</h2>
      <p>
        Mastering Git involves understanding and applying best practices that
        can help manage your projects more efficiently. This guide provides
        insights into some critical aspects of using Git effectively.
      </p>
      {/* Good Practices in Software Development */}
      <Row className="mt-4">
        <Col>
          <h3 id="good-practices">Good Practices in Software Development</h3>
          <p>
            Adopting good software development practices is crucial for the
            success of any project. Here are some key practices:
          </p>
          <ul>
            <li>
              <strong>Define and Follow a Workflow:</strong> Establish and
              adhere to a workflow like Git Flow or feature branching to
              streamline development and collaboration.
            </li>
            <li>
              <strong>Make Small Changes:</strong> Smaller, incremental changes
              are easier to manage and review than large overhauls.
            </li>
            <li>
              <strong>Release Often:</strong> Regular releases help to iterate
              quickly and respond to feedback effectively.
            </li>
            <li>
              <strong>Utilize Pull Requests:</strong> Use pull requests to
              initiate code reviews and merge changes, ensuring quality and
              shared understanding of the codebase.
            </li>
            <li>
              <strong>Conduct Thorough Reviews:</strong> Peer reviews of code
              are essential for maintaining code quality and reducing bugs in
              production.
            </li>
          </ul>
        </Col>
      </Row>
      {/* Merge vs. Rebase */}
      {/* <Row>
        <Col>
          <h3 id="merge-vs-rebase">Merge vs. Rebase</h3>
          <p>
            Both `git merge` and `git rebase` are used to integrate changes from
            one branch into another, but they do it in different ways:
          </p>
          <ul>
            <li>
              <strong>Merge:</strong> Merging creates a new "merge commit" in
              your history. It preserves the history of the project by
              maintaining the timeline of both branches.
            </li>
            <li>
              <strong>Rebase:</strong> Rebasing rewrites the project history by
              placing your branch's changes on top of the current branch.
              Rebasing makes a cleaner project history.
            </li>
          </ul>
          <CodeBlock code={`git merge feature-branch\n git rebase master`} />
          <p>
            Use merging to avoid altering the project history and rebase for a
            cleaner and linear history.
          </p>
        </Col>
      </Row> */}
      {/* Git Tagging */}
      <Row className="mt-4">
        <Col>
          <h3 id="git-tagging">Using Git Tags</h3>
          <p>
            Tags in Git are used to create stable releases or to mark a specific
            point in your repository's history as important.
          </p>
          <ul>
            <li>
              <strong>Creating a Tag:</strong> Use `git tag {"<tagname>"}` to
              create a lightweight tag, or `git tag -a {"<tagname>"} -m
              "message"` for an annotated tag.
            </li>
            <li>
              <strong>Listing Tags:</strong> Use `git tag` to list all tags in
              the repository.
            </li>
            <li>
              <strong>Pushing Tags:</strong> Tags do not automatically transfer
              to the remote repository. Use `git push origin {"<tagname>"}` to
              push a single tag or `git push origin --tags` to push all tags.
            </li>
          </ul>
          <CodeBlock
            code={`git tag v1.0
git tag -a v1.1 -m "Release version 1.1"
git push origin v1.1`}
          />
          <p>
            Tags can help you and your team to refer to specific releases
            without having to remember commit hashes.
          </p>
        </Col>
      </Row>

      {/* Using Git Stash */}
      <Row className="mt-4">
        <Col>
          <h3 id="using-git-stash">Using Git Stash</h3>
          <p>
            Git stash temporarily shelves (or stashes) changes you've made to
            your working directory, allowing you to work on something else, and
            then come back and re-apply them later on.
          </p>
          <CodeBlock
            code={`git stash
git stash apply`}
          />
          <p>
            Stashing is handy if you need to quickly switch context and work on
            something else, but you're not ready to commit.
          </p>
        </Col>
      </Row>

      {/* CI/CD and GitHub Actions */}
      <Row className="mt-4">
        <Col>
          <h3 id="ci-cd-github-actions">CI/CD and GitHub Actions</h3>
          <p>
            Continuous Integration (CI) and Continuous Deployment (CD) are
            practices that automate the integration of code changes and the
            deployment of your application:
          </p>
          <ul>
            <li>
              <strong>CI:</strong> Automatically test and merge code changes.
            </li>
            <li>
              <strong>CD:</strong> Automatically deploy code changes to a
              production or staging environment.
            </li>
          </ul>
          <p>
            GitHub Actions makes CI/CD easy with workflows that can handle
            build, test, and deployment tasks directly within your GitHub
            repository.
          </p>
          <CodeBlock
            language=""
            code={`name: CI\n on: push\n jobs: build\n runs-on: ubuntu-latest\n steps:\n - uses: actions/checkout@v2\n - name: Run a one-line script\n run: python tests.py`}
          />
        </Col>
      </Row>
    </Container>
  );
};

export default BestPracticesAndResources;
