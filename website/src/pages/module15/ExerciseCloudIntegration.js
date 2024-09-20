import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const ExerciseCloudIntegration = () => {
  const exerciseLinks = [
    // Add links to other exercises as needed
  ];

  const location = useLocation();
  const module = 15;
  return (
    <ModuleFrame
      module={module}
      isCourse={false}
      title="Module 15: Exercise Cloud Integration (with GCP)"
      courseLinks={exerciseLinks}
    >
      {location.pathname === `/module${module}/exercise` && (
        <>
          <Row>
            <p>
              In this module, you will practice integrating your applications
              with Google Cloud Platform (GCP).
            </p>
          </Row>
          <Row>
            <Col>
              <p>Last Updated: {"2024-09-20"}</p>
            </Col>
          </Row>
        </>
      )}
      <Row>
        <Col md={11}>
          <DynamicRoutes routes={exerciseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default ExerciseCloudIntegration;
