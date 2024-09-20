import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const SettingUpPythonEnvironment = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Setting Up a Python Environment</h1>
      <p>
        Setting up a dedicated Python environment for your projects can help
        manage dependencies and ensure that different projects can run
        independently on the same machine. This section will guide you through
        the setup of a Python virtual environment using <code>venv</code>, a
        built-in tool for creating isolated Python environments.
      </p>
      <Row>
        <Col md={12}>
          <h3 id="create-environment">Create a Virtual Environment</h3>
          <p>
            To create a new virtual environment, use the <code>venv</code>{" "}
            module, which is included with Python 3.3 and above.
          </p>
          <CodeBlock code={`python3 -m venv myenv`} language="bash" />
          <p>
            This command creates a new directory called <code>myenv</code> that
            contains a fresh, isolated Python installation. You can replace
            "myenv" with a name of your choice for your environment.
          </p>

          <h3 id="activate-environment">Activate the Virtual Environment</h3>
          <p>
            Once the virtual environment is created, you need to activate it:
          </p>
          <p>
            On <strong>Windows</strong>:
          </p>
          <CodeBlock code={`.\\myenv\\Scripts\\activate`} language="bash" />
          <p>
            On <strong>Unix or MacOS</strong>:
          </p>
          <CodeBlock code={`source myenv/bin/activate`} language="bash" />
          <p>
            After activation, your command prompt will change to indicate that
            you are now working inside the virtual environment.
          </p>

          <h3 id="deactivate-environment">
            Deactivate the Virtual Environment
          </h3>
          <p>
            To deactivate the virtual environment and return to the global
            Python environment, use:
          </p>
          <CodeBlock code={`deactivate`} language="bash" />
          <p>
            This command deactivates the environment, and your command prompt
            will return to its normal state.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default SettingUpPythonEnvironment;
