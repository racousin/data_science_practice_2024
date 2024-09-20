import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseCloudIntegration = () => {
  const courseLinks = []
  // const courseLinks = [
    // {
    //   to: "/introduction",
    //   label: "Introduction to Cloud Computing and GCP",
    //   component: lazy(() => import("pages/module15/course/Introduction")),
    // },
    // {
    //   to: "/managing-compute-resources",
    //   label: "Managing Compute Resources",
    //   component: lazy(() =>
    //     import("pages/module15/course/ManagingComputeResources")
    //   ),
    // },
    // {
    //   to: "/storing-managing-data",
    //   label: "Storing and Managing Data",
    //   component: lazy(() =>
    //     import("pages/module15/course/StoringManagingData")
    //   ),
    // },
    // {
    //   to: "/data-analysis-machine-learning",
    //   label: "Data Analysis and Machine Learning on GCP",
    //   component: lazy(() =>
    //     import("pages/module15/course/DataAnalysisMachineLearning")
    //   ),
    // },
    // {
    //   to: "/networking-security",
    //   label: "Networking and Security",
    //   component: lazy(() => import("pages/module15/course/NetworkingSecurity")),
    // },
    // {
    //   to: "/devops-in-the-cloud",
    //   label: "DevOps in the Cloud",
    //   component: lazy(() => import("pages/module15/course/DevOpsInTheCloud")),
    // },
    // {
    //   to: "/architecting-scalable-applications",
    //   label: "Architecting Scalable Applications",
    //   component: lazy(() =>
    //     import("pages/module15/course/ArchitectingScalableApplications")
    //   ),
    // },
  // ];

  const location = useLocation();
  const module = 15;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 15: Cloud Integration (with GCP)"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about integrating your applications
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
          <DynamicRoutes routes={courseLinks} />
        </Col>
      </Row>
    </ModuleFrame>
  );
};

export default CourseCloudIntegration;
