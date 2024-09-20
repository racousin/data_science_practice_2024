import React, { useEffect, useState } from "react";
import { Nav, Button, Container } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import EvaluationModal from "components/EvaluationModal";

const ModuleNavigation = ({ module, isCourse, title = "" }) => {
  const [navBarHeight, setNavBarHeight] = useState(0); // Set default height or get dynamically
  const navigate = useNavigate();

  useEffect(() => {
    const updateNavBarHeight = () => {
      const navbar = document.querySelector(".navbar");
      if (navbar) {
        setNavBarHeight(navbar.offsetHeight); // Update only if navbar exists
      }
    };

    // Update on component mount
    updateNavBarHeight();

    // Ensure the height is updated on window resize
    window.addEventListener("resize", updateNavBarHeight);

    // Cleanup listener on component unmount
    return () => window.removeEventListener("resize", updateNavBarHeight);
  }, []);

  const navigateTo = (path) => {
    navigate(path);
  };

  return (
    <Nav
      className="justify-content-between align-items-center navigation-header"
      style={{ top: `${navBarHeight}px` }}
    >
      <h1 className="module-title">{title}</h1>
      <div>
        {module > 0 && (
          <Button
            variant="outline-primary"
            className="nav-button button-outline"
            onClick={() => navigateTo(`/module${module - 1}/course`)}
          >
            Previous Module
          </Button>
        )}
        {isCourse !== null &&
          (isCourse ? (
            <Button
              variant="outline-secondary"
              className="nav-button button-outline"
              onClick={() => navigateTo(`/module${module}/exercise`)}
            >
              Exercises
            </Button>
          ) : (
            <>
              <Button
                variant="outline-secondary"
                className="nav-button button-outline"
                onClick={() => navigateTo(`/module${module}/course`)}
              >
                Courses
              </Button>
              <EvaluationModal module={module} />
            </>
          ))}

        {module < 14 && (
          <Button
            variant="outline-success"
            className="nav-button button-outline"
            onClick={() => navigateTo(`/module${module + 1}/course`)}
          >
            Next Module
          </Button>
        )}
      </div>
    </Nav>
  );
};

export default ModuleNavigation;
