import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const InstallPython = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Installing Python</h1>
      <Row>
        <Col md={12}>
          <h3 id="windows">Windows</h3>
          <p>Follow these steps to install Python on Windows:</p>
          <ol>
            <li>
              Download the latest version of Python from the official Python
              website:
              <br />
              <a
                href="https://www.python.org/downloads/windows/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Python Downloads for Windows
              </a>
            </li>
            <li>
              Run the downloaded installer. Ensure to select "Add Python 3.x to
              PATH" at the start of the installation process.
            </li>
            <li>Follow the installation prompts to complete the setup.</li>
          </ol>
          <h3 id="mac">MacOS</h3>
          <p>
            MacOS comes with Python pre-installed. To check the installed
            version:
          </p>
          <ol>
            <li>
              Open a terminal window and type the following command to check
              your current Python version:
              <CodeBlock code={`python3 --version`} />
              <CodeBlock
                showCopy={false}
                code={`$ python3 --version
Python 3.10.12
`}
                language=""
              />
              or try:
              <CodeBlock code={`python --version`} />
            </li>
            <li>
              If you need a newer version, consider installing Python via{" "}
              <a
                href="https://brew.sh/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Homebrew
              </a>{" "}
              or directly from the Python website.
            </li>
          </ol>
          <h3 id="linux">Linux</h3>
          <p>
            Most Linux distributions come with Python pre-installed. To verify
            or install Python, you can use your distribution's package manager:
          </p>
          <ol>
            <li>
              Open a terminal window and check the installed version of Python:
              <CodeBlock code={`python3 --version`} />
              <CodeBlock
                showCopy={false}
                code={`$ python3 --version
Python 3.10.12
`}
                language=""
              />
              or try:
              <CodeBlock code={`python --version`} />
            </li>
            <li>
              If Python is not installed, or if you need a different version,
              use your distribution’s package manager to install Python. For
              example, on Ubuntu, you would use:
              <CodeBlock code={`sudo apt-get install python3`} />
            </li>
          </ol>
        </Col>
      </Row>
      <Row>
        <Col md={12}>
          <h3 id="install-pip">Install pip</h3>
          <p>
            <code>pip</code> is Python's package installer and is included by
            default with Python versions 3.4 and above. It's crucial for
            managing third-party Python packages. Here’s how to ensure it is
            installed and up to date:
          </p>
          <ol>
            <li>
              To check if <code>pip</code> is installed, open a terminal or
              command prompt and type:
              <CodeBlock code={`pip --version`} />
              <CodeBlock
                showCopy={false}
                code={`$ pip --version
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
`}
                language=""
              />
            </li>

            <li>
              If <code>pip</code> is not installed, you can install it by
              downloading <code>get-pip.py</code>:
              <br />
              <a
                href="https://bootstrap.pypa.io/get-pip.py"
                target="_blank"
                rel="noopener noreferrer"
              >
                Download get-pip.py
              </a>
            </li>
            <li>
              After downloading, run the following command in the directory
              where <code>get-pip.py</code> is located:
              <CodeBlock code={`python get-pip.py`} />
            </li>
            <li>
              To upgrade an existing <code>pip</code> installation to the latest
              version, use:
              <CodeBlock code={`pip install --upgrade pip`} />
            </li>
          </ol>
          <p>
            Ensuring <code>pip</code> is installed and up to date allows you to
            easily manage and install packages, which are often necessary for
            development projects.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default InstallPython;
