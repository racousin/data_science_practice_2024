import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CoursePython = () => {
  const courseLinks = [
    {
      to: "/introduction",
      label: "Introduction",
      component: lazy(() => import("pages/module2/course/Introduction")),
    },
    {
      to: "/install-python",
      label: "Install Python",
      component: lazy(() => import("pages/module2/course/InstallPython")),
      subLinks: [
        { id: "mac", label: "Mac" },
        { id: "windows", label: "Windows" },
        { id: "linux", label: "Linux" },
        { id: "install-pip", label: "Install pip" },
      ],
    },
    {
      to: "/setting-up-python-environment",
      label: "Setting Up Python Environment",
      component: lazy(() =>
        import("pages/module2/course/SettingUpPythonEnvironment")
      ),
      subLinks: [
        { id: "create-environment", label: "Create a Virtual Environment" },
        {
          id: "activate-environment",
          label: "Activate the Virtual Environment",
        },
        {
          id: "deactivate-environment",
          label: "Deactivate the Virtual Environment",
        },
      ],
    },
    {
      to: "/installing-packages",
      label: "Installing Packages",
      component: lazy(() => import("pages/module2/course/InstallingPackages")),
      subLinks: [
        { id: "install-package", label: "Install a Package" },
        {
          id: "install-specific-version",
          label: "Install a Specific Version of a Package",
        },
        {
          id: "install-from-requirements",
          label: "Install Packages from a Requirements File",
        },
      ],
    },
    {
      to: "/building-packages",
      label: "Building Packages",
      component: lazy(() => import("pages/module2/course/BuildingPackages")),
      subLinks: [
        {
          id: "package-structure",
          label: "Create a New Directory for Your Package",
        },
        {
          id: "setup-file",
          label: "Create the `setup.py` File",
        },
        {
          id: "package-directory",
          label: "Create the Package Directory",
        },
        {
          id: "add-modules",
          label: "Add Modules to Your Package",
        },
        {
          id: "build-package",
          label: "Build the Package",
        },
        {
          id: "editable-install",
          label: "Editable Install",
        },
        {
          id: "publishing-package",
          label: "Publishing Your Package",
        },
      ],
    },

    {
      to: "/best-practices",
      label: "Best Practices And Ressources",
      component: lazy(() =>
        import("pages/module2/course/BestPracticesAndRessources")
      ),
      subLinks: [
        {
          id: "other-package-managers",
          label: "Other Package Managers",
        },
        {
          id: "testing-and-unit-tests",
          label: "Testing and Unit Tests",
        },
        {
          id: "syntax-and-linting",
          label: "Syntax and Linting",
        },
        {
          id: "resources",
          label: "Additional Resources",
        },
      ],
    },
  ];
  const location = useLocation();
  const module = 2;
  return (
    <ModuleFrame
      module={2}
      isCourse={true}
      title="Module 2: Python Environment and Package"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row></Row>
          <Row>
            <Col>
              <p>Last Updated: {"2024-09-20"}</p>
            </Col>
          </Row>
        </>
      )}
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CoursePython;
