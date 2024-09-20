import React from "react";
import { Container, Row, Col, Button } from "react-bootstrap";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { a11yDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import CodeBlock from "components/CodeBlock";

const InstallingGit = () => {
  const commands = {
    mac: "brew install git",
    windows: "choco install git",
    linux: "sudo apt install git",
    version: "git --version",
  };

  return (
    <Container fluid>
      <h2>Installing Git</h2>
      <p>
        To install Git on your computer, you can use a package manager specific
        to your operating system, or download the installer directly from the{" "}
        <a
          href="https://git-scm.com/downloads"
          target="_blank"
          rel="noopener noreferrer"
        >
          official Git website
        </a>
        .
      </p>
      <Row>
        <Col>
          <h3 id="mac">Mac</h3>
          <p>
            If you have{" "}
            <a
              href="https://brew.sh/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Homebrew
            </a>{" "}
            installed on your Mac, you can install Git by running the following
            command in the Terminal:
          </p>
          <CodeBlock code={commands.mac} />
          <p>
            Don't have Homebrew?{" "}
            <a
              href="https://brew.sh/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Install Homebrew here
            </a>
            .
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3 id="windows">Windows</h3>
          <p>
            For Windows users,{" "}
            <a
              href="https://chocolatey.org/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Chocolatey
            </a>{" "}
            can be used to install Git easily:
          </p>
          <CodeBlock code={commands.windows} />
          <p>
            Alternatively, download the Git installer directly from the{" "}
            <a
              href="https://git-scm.com/download/win"
              target="_blank"
              rel="noopener noreferrer"
            >
              Git website
            </a>{" "}
            and follow the installation instructions.
          </p>
          <p>
            Don't have Chocolatey?{" "}
            <a
              href="https://chocolatey.org/install"
              target="_blank"
              rel="noopener noreferrer"
            >
              Install Chocolatey here
            </a>
            .
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3 id="linux">Linux</h3>
          <p>
            If you are using a Debian-based Linux distribution, such as Ubuntu,
            you can install Git using the following command in your terminal:
          </p>
          <CodeBlock code={commands.linux} />
          <p>
            For other Linux distributions, you can find specific installation
            instructions on the{" "}
            <a
              href="https://git-scm.com/download/linux"
              target="_blank"
              rel="noopener noreferrer"
            >
              Git website
            </a>
            .
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Validate Your Installation</h3>
          <p>
            To verify that Git has been installed correctly, open a terminal or
            command prompt and type the following command:
          </p>
          <CodeBlock code={commands.version} />
          <CodeBlock
            code={`$ git --version
git version 2.34.1`}
            showCopy={false}
            language=""
          />
          <p>
            This command should display the installed version of Git, confirming
            that the software is ready for use.
          </p>
        </Col>
      </Row>
      <Row className="mt-4">
        <Col>
          <h3>Troubleshooting</h3>
          <p>
            If you encounter any issues during installation, consult the{" "}
            <a
              href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git"
              target="_blank"
              rel="noopener noreferrer"
            >
              Git installation guide
            </a>
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default InstallingGit;
