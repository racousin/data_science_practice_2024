import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const InstallingPackages = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Installing Packages</h1>
      <Row>
        <Col>
          <h3 id="install-package">Install a Package</h3>
          <p>To install a package using pip, use the following command:</p>
          <CodeBlock code={`pip install numpy`} />
          <p>
            This command installs the latest version of the specified package,
            in this case, numpy.
          </p>

          <h3 id="install-specific-version">
            Install a Specific Version of a Package
          </h3>
          <p>
            To install a specific version of a package, use the following
            command:
          </p>
          <CodeBlock code={`pip install numpy==1.19.5`} />
          <p>
            This command installs version 1.19.5 of the numpy package.
            Specifying the version is useful when you need a particular version
            that is compatible with your code.
          </p>

          <h3 id="install-from-requirements">
            Install Packages from a Requirements File
          </h3>
          <p>
            To install packages listed in a requirements file, use the following
            command:
          </p>
          <CodeBlock code={`pip install -r requirements.txt`} />
          <p>
            A requirements file lists all the packages your project depends on,
            along with their versions. This command reads the requirements file
            and installs all the listed packages.
          </p>
          <h4>Example of a requirements file:</h4>
          <CodeBlock code={`numpy==1.19.5\npandas==1.1.5\nscipy==1.5.4`} />
          <p>
            This example requirements file specifies exact versions for numpy,
            pandas, and scipy. Using a requirements file ensures consistency
            across different environments.
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default InstallingPackages;
