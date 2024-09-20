import React, { lazy } from "react";
import { Container, Row, Col } from "react-bootstrap";
import DynamicRoutes from "components/DynamicRoutes";
import ModuleFrame from "components/ModuleFrame";
import { useLocation } from "react-router-dom";

const CourseDocker = () => {
  const courseLinks = []
  // const courseLinks = [
  //   {
  //     to: "/introduction",
  //     label: "Introduction to Containerization",
  //     component: lazy(() => import("pages/module14/course/Introduction")),
  //     subLinks: [
  //       { id: "what-is-containerization", label: "What is Containerization?" },
  //       {
  //         id: "benefits-of-containerization",
  //         label: "Benefits of Containerization",
  //       },
  //       { id: "containers-vs-vms", label: "Containers vs. Virtual Machines" },
  //       { id: "docker-overview", label: "Docker Overview" },
  //     ],
  //   },
  //   {
  //     to: "/docker-fundamentals",
  //     label: "Docker Fundamentals",
  //     component: lazy(() => import("pages/module14/course/DockerFundamentals")),
  //     subLinks: [
  //       { id: "docker-architecture", label: "Docker Architecture" },
  //       {
  //         id: "docker-objects",
  //         label: "Docker Objects (Images, Containers, Volumes, Networks)",
  //       },
  //       { id: "dockerfile-basics", label: "Dockerfile Basics" },
  //       { id: "docker-cli", label: "Docker CLI Commands" },
  //     ],
  //   },
  //   {
  //     to: "/working-with-containers",
  //     label: "Working with Docker Containers",
  //     component: lazy(() =>
  //       import("pages/module14/course/WorkingWithContainers")
  //     ),
  //     subLinks: [
  //       { id: "running-containers", label: "Running Containers" },
  //       { id: "managing-containers", label: "Managing Containers" },
  //       { id: "container-lifecycle", label: "Container Lifecycle" },
  //       { id: "container-networking", label: "Container Networking" },
  //       { id: "data-management", label: "Data Management and Volumes" },
  //     ],
  //   },
  //   {
  //     to: "/docker-compose-services",
  //     label: "Docker Compose and Services",
  //     component: lazy(() =>
  //       import("pages/module14/course/DockerComposeServices")
  //     ),
  //     subLinks: [
  //       { id: "docker-compose-intro", label: "Introduction to Docker Compose" },
  //       {
  //         id: "compose-file-structure",
  //         label: "Docker Compose File Structure",
  //       },
  //       {
  //         id: "multi-container-apps",
  //         label: "Creating Multi-Container Applications",
  //       },
  //       { id: "docker-services", label: "Docker Services and Swarm Mode" },
  //     ],
  //   },
  //   {
  //     to: "/docker-for-data-science",
  //     label: "Docker for Data Science and Machine Learning",
  //     component: lazy(() =>
  //       import("pages/module14/course/DockerForDataScience")
  //     ),
  //     subLinks: [
  //       {
  //         id: "data-science-workflows",
  //         label: "Containerizing Data Science Workflows",
  //       },
  //       {
  //         id: "ml-model-deployment",
  //         label: "Deploying Machine Learning Models",
  //       },
  //       { id: "gpu-support", label: "GPU Support in Docker" },
  //       {
  //         id: "jupyter-notebooks",
  //         label: "Running Jupyter Notebooks in Docker",
  //       },
  //     ],
  //   },
  // ];

  const location = useLocation();
  const module = 14;
  return (
    <ModuleFrame
      module={module}
      isCourse={true}
      title="Module 14: Docker"
      courseLinks={courseLinks}
    >
      {location.pathname === `/module${module}/course` && (
        <>
          <Row>
            <p>
              In this module, you will learn about Docker, a platform for
              building, shipping, and running applications in containers.
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

export default CourseDocker;
