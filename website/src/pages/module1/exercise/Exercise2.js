import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Exercise2 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">
        Exercise 2: Managing Branches and Correcting Errors
      </h1>
      <Row>
        <Col md={12}>
          <h2>Context</h2>
          <p>
            You are writing technical documentation for a mathematics library. You've started with a document explaining basic arithmetic operations, but you accidentally made an error in one of the examples. As you continue working, you'll discover this error and need to fix it while managing your ongoing work.
          </p>

          <h2>Instructions</h2>
          <ol>
            <li>
              <h3>Initial Setup</h3>
              <p>Create a new repository on GitHub named "math-docs" and clone it to your local machine.</p>
              <CodeBlock
                code={`$ git clone https://github.com/your-username/math-docs.git
$ cd math-docs`}
                language="bash"
              />
            </li>

            <li>
              <h3>Create and Push Initial Content</h3>
              <p>Create a README.md file in your local repository with the following content:</p>
              <CodeBlock
                code={`# Mathematics Library Documentation

## 1. Basic Arithmetic

### 1.1 Addition
The sum of two numbers a and b is represented as a + b.
Example: 2 + 3 = 5

### 1.2 Multiplication
The product of two numbers a and b is represented as a * b.
Example: 2 * 3 = 5`}
                language="markdown"
              />
              <p>Add this file to git, commit it with the message "Add initial documentation on basic arithmetic", and push it to the main branch on GitHub.</p>
              <CodeBlock
                code={`$ git add README.md
$ git commit -m "Add initial documentation on basic arithmetic"
$ git push origin main`}
                language="bash"
              />
            </li>

            <li>
              <h3>Start Working on Advanced Operations</h3>
              <p>Create and switch to a new branch called "advanced-operations". Add the following content to the end of your README.md file:</p>
              <CodeBlock
                code={`
## 2. Advanced Operations

### 2.1 Exponentiation
Exponentiation is represented as a^b or pow(a, b).
Example: 2^3 = 8`}
                language="markdown"
              />
              <p>As you're adding this content, you notice the error in the multiplication example. Your colleague points out that people might think you don't know how to count. You realize you need to fix this error before continuing with the advanced operations.</p>
              <p>Commit your changes to the "advanced-operations" branch with the message "Start documentation on advanced operations".</p>
              <CodeBlock
                code={`$ git checkout -b advanced-operations
$ git add README.md
$ git commit -m "Start documentation on advanced operations"`}
                language="bash"
              />
            </li>

            <li>
              <h3>Correct the Error in Basic Arithmetic</h3>
              <p>Switch back to the main branch and create a new branch called "fix-multiplication-error". In the README.md file, correct the multiplication example to read:</p>
              <CodeBlock
                code={`Example: 2 * 3 = 6`}
                language="markdown"
              />
              <p>Commit this change with the message "Fix multiplication example in basic arithmetic" and push this branch to GitHub. Create a pull request to merge these changes into the main branch. Once approved, merge the pull request.</p>
              <CodeBlock
                code={`$ git checkout main
$ git checkout -b fix-multiplication-error
$ git add README.md
$ git commit -m "Fix multiplication example in basic arithmetic"
$ git push origin fix-multiplication-error`}
                language="bash"
              />
            </li>

            <li>
              <h3>Integrate Changes into Advanced Operations Branch</h3>
              <p>Now that the error is fixed in the main branch, switch back to your "advanced-operations" branch. Merge the changes from the main branch into this branch. This will bring in the corrected multiplication example while keeping your work on advanced operations.</p>
              <p>Commit these merged changes and push the updated "advanced-operations" branch to GitHub.</p>
              <CodeBlock
                code={`$ git checkout advanced-operations
$ git merge main
$ git add README.md
$ git commit -m "Merge corrected basic arithmetic into advanced operations"
$ git push origin advanced-operations`}
                language="bash"
              />
            </li>
          </ol>

          <h2>Verification</h2>
          <p>To verify your work, review the content of your README.md file in the "advanced-operations" branch. It should contain:</p>
          <ul>
            <li>The correct multiplication example (2 * 3 = 6)</li>
            <li>The new section on advanced operations, including exponentiation</li>
          </ul>
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise2;