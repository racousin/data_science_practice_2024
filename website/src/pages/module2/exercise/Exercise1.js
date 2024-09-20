import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import EvaluationModal from "components/EvaluationModal";

const Exercise1 = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Exercise 1: Creating a Python Package</h1>
      <p>
        In this exercise, you will create a Python package named `mysupertools`
        with a module that contains a function to multiply two values. This
        function will return the product if both arguments are numbers, and the
        string "error" otherwise.
      </p>
      <Row>
        <Col>
          <h2>Instructions</h2>
          <ol>
            <li>Your final directory structure should look like this:</li>
            <CodeBlock
              code={`mysupertools/\n├── setup.py\n└── mysupertools/\n    ├── __init__.py\n    └── tool/\n        ├── __init__.py\n        └── multiplication_a_b.py`}
            />
            <li>
              The <code>multiplication_a_b.py</code> file contains a function{" "}
              <code>multiply(a, b)</code> that return <i>a x b</i> results if
              doable or "error".
            </li>
            <li>
              Create a <code>setup.py</code> file in the{" "}
              <code>mysupertools</code> directory with the necessary content to
              make it a package
            </li>
            <CodeBlock
              code={`from setuptools import setup, find_packages

setup(
    name='mysupertools',
    version='0.1',
    packages=find_packages(),
    ...
)
`}
              language={"python"}
            />
            <li>
              Don't forget the<code>__init__.py</code> files inside both the{" "}
              <code>mysupertools</code> and <code>tool</code> directories to
              make them Python packages:
            </li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col>
          <h2>Testing Your Code</h2>
          <p>
            To ensure your package is working correctly, follow these steps to
            test your code:
          </p>
          <ol>
            <li>
              First, install your package. Ideally, do this in a new virtual
              environment:
            </li>
            <CodeBlock code={`pip install $username/module2/mysupertools`} />
            <li>Next, open a Python session and import your function:</li>
            <CodeBlock code={`python`} />
            <CodeBlock
              code={`from mysupertools.tool.multiplication_a_b import multiply
                `}
              language={"python"}
            />
            <li>Finally, test the functionality of your function:</li>
            <CodeBlock
              code={`assert multiply(4, 5) == 20`}
              language={"python"}
            />
            <CodeBlock
              code={`assert multiply("a", 5) == "error"`}
              language={"python"}
            />
          </ol>
        </Col>
      </Row>
      <Row>
        <Col>
          <EvaluationModal module={module} />
        </Col>
      </Row>
    </Container>
  );
};

export default Exercise1;
