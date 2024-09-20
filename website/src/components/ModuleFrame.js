import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import ModuleNavigation from "components/ModuleNavigation";
import NavigationMenu from "components/NavigationMenu";
import DynamicRoutes from "./DynamicRoutes";

const ModuleFrame = ({ module, isCourse, title, children, courseLinks }) => {
  return (
    <Row>
      <ModuleNavigation module={module} isCourse={isCourse} title={title} />
      <Col md={3}>
        <NavigationMenu
          links={courseLinks}
          prefix={`/module${module}/${isCourse ? "course" : "exercise"}`}
        />
      </Col>
      <Col md={9} className="module-content">
        {children}
      </Col>
    </Row>
  );
};

export default ModuleFrame;
