import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const BuildingPackages = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Building Python Packages</h1>
      <Row>
        <Col>
          <h3 id="package-structure">Package Project Structure</h3>
          <p>Here's what the directory structure should look like:</p>
          <pre>
            {`
mypackage/
├── mypackage/
│   ├── __init__.py
│   ├── module1.py
│   ├── module2.py
│   ├── tools/
│       ├── __init__.py
│       └── tools1.py
├── setup.py
└── README.md
      `}
          </pre>
          <p>Here's a brief explanation of each component:</p>
          <ul>
            <li>
              <strong>mypackage/:</strong> The root directory of your package
              project.
            </li>
            <li>
              <strong>mypackage/:</strong> The directory containing your
              package's modules.
            </li>
            <li>
              <strong>__init__.py:</strong> A file that initializes the package
              and can contain package-level variables and imports.
            </li>
            <li>
              <strong>module1.py:</strong> A module within the package. You can
              add more modules as needed.
            </li>
            <li>
              <strong>module2.py:</strong> Another module within the package.
            </li>
            <li>
              <strong>setup.py:</strong> The setup script containing metadata
              about the package and instructions on how to install it.
            </li>
            <li>
              <strong>README.md:</strong> A README file providing an overview of
              the package, installation instructions, and usage examples.
            </li>
          </ul>
        </Col>
      </Row>

      <Row>
        <Col>
          <h3 id="create-directory">Create a New Directory for Your Package</h3>
          <p>First, create a new directory where your package will reside:</p>
          <CodeBlock code={`mkdir mypackage`} />
          <p>Navigate into the newly created directory:</p>
          <CodeBlock code={`cd mypackage`} />
        </Col>
      </Row>
      <Row>
        <Col>
          <h3 id="setup-file">Create the `setup.py` File</h3>
          <p>
            The `setup.py` file is the center of a Python project. It contains
            metadata about the package and instructions on how to install it.
          </p>
          <p>Create the `setup.py` file:</p>
          <CodeBlock code={`touch setup.py`} />
          <p>Open `setup.py` in a text editor and add the following code:</p>
          <CodeBlock
            code={`from setuptools import setup

setup(
    name='mypackage',
    version='1.0',
    description='A simple Python package',
    author='Your Name',
    author_email='youremail@example.com',
    url='https://github.com/yourusername/mypackage',
    packages=['mypackage'],
)`}
            language={"python"}
          />
          <p>In this setup script:</p>
          <ul>
            <li>
              <strong>name:</strong> The name of your package.
            </li>
            <li>
              <strong>version:</strong> The current version of your package.
            </li>
            <li>
              <strong>description:</strong> A brief description of your package.
            </li>
            <li>
              <strong>author:</strong> Your name.
            </li>
            <li>
              <strong>author_email:</strong> Your email address.
            </li>
            <li>
              <strong>url:</strong> The URL to the homepage of the package
              (usually a GitHub repository).
            </li>
            <li>
              <strong>packages:</strong> A list of all Python import packages
              that should be included in the distribution package.
            </li>
          </ul>
        </Col>
      </Row>
      <Row>
        <Col>
          <h3 id="package-directory">Create the Package Directory</h3>
          <p>Create a new directory with the same name as your package:</p>
          <CodeBlock code={`mkdir mypackage`} />
          <p>
            Inside this directory, create an `__init__.py` file to indicate that
            this directory should be treated as a package:
          </p>
          <CodeBlock code={`touch mypackage/__init__.py`} />
          <p>
            The `__init__.py` file can be empty, or you can put initialization
            code for your package there. This file tells Python that the
            directory should be treated as a package, allowing you to import
            modules from it.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h3 id="add-modules">Add Modules to Your Package</h3>
          <p>
            Create Python modules within the `mypackage` directory. For example:
          </p>
          <CodeBlock code={`touch mypackage/module1.py mypackage/module2.py`} />
          <p>
            Add your code to these modules. For example, `module1.py` might
            contain:
          </p>
          <CodeBlock
            code={`def hello():
    print("Hello from module1!")`}
            language={"python"}
          />
          <p>And `module2.py` might contain:</p>
          <CodeBlock
            code={`def goodbye():
    print("Goodbye from module2!")`}
            language={"python"}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <h3 id="build-package">Build the Package</h3>
          <p>Use the following command to build your package:</p>
          <CodeBlock code={`python setup.py sdist`} />
          <p>
            This command creates a source distribution (sdist) containing your
            package. The distribution file is stored in the `dist` directory.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h3 id="install-package">Install the Package</h3>
          <p>Install the newly created package using pip:</p>
          <CodeBlock code={`pip install dist/mypackage-1.0.tar.gz`} />
          <p>
            Replace `mypackage-1.0.tar.gz` with the actual name of the generated
            file.
          </p>
          <p>
            Now you can use your package in Python just like any other package:
          </p>
          <CodeBlock
            code={`import mypackage.module1 as mod1
import mypackage.module2 as mod2

mod1.hello()
mod2.goodbye()`}
            language={"python"}
          />
        </Col>
      </Row>
      <Row>
        <Col>
          <h3 id="editable-install">Editable Install</h3>
          <p>
            During development, it’s useful to install your package in an
            "editable" mode. This allows you to modify your package without
            reinstalling it every time. Use the following command to install
            your package in editable mode:
          </p>
          <CodeBlock code={`pip install -e .`} />
          <p>
            This installs the package in the current directory and links it to
            your Python environment. Changes to your package will be reflected
            immediately without needing to reinstall.
          </p>
        </Col>
      </Row>
      <Row>
        <Col>
          <h3 id="publishing-package">Publishing Your Package</h3>
          <p>
            Once your package is ready, you might want to share it with the
            world by publishing it to the Python Package Index (PyPI). Here are
            the basic steps:
          </p>
          <ol>
            <li>Install Twine, a tool for publishing Python packages:</li>
            <CodeBlock code={`pip install twine`} />
            <li>Upload your package to PyPI:</li>
            <CodeBlock code={`twine upload dist/*`} />
            <p>
              You'll need a PyPI account and will be prompted to enter your
              credentials.
            </p>
          </ol>
          <p>
            For detailed instructions on how to publish your package, visit the{" "}
            <a
              href="https://packaging.python.org/tutorials/packaging-projects/"
              target="_blank"
              rel="noopener noreferrer"
            >
              official Python packaging guide
            </a>
            .
          </p>
        </Col>
      </Row>
    </Container>
  );
};

export default BuildingPackages;
