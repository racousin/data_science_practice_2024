import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import {
  Container,
  Accordion,
  Card,
  Badge,
  ListGroup,
  Row,
  Col,
} from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCheckCircle,
  faTimesCircle,
  faChevronDown,
  faChevronUp,
} from "@fortawesome/free-solid-svg-icons";
import ArrayProgress from "components/ArrayProgress";
import BackButton from "components/BackButton";
import OverallProgress from "components/OverallProgress";
import { format, parseISO } from "date-fns";

const Student = () => {
  const { repositoryId, studentId } = useParams();
  const [modulesResults, setModulesResults] = useState({});
  const [error, setError] = useState("");
  const [activeKey, setActiveKey] = useState(null);
  const [overviewProgress, setOverviewProgress] = useState(0);
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date
      .toLocaleString("en-US", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false, // Use 24-hour time
      })
      .replace(",", ""); // Remove the comma after the date
  };
  useEffect(() => {
    fetch(`/repositories/${repositoryId}/students/${studentId}.json`)
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to load the data");
        }
        return response.json();
      })
      .then((data) => {
        setModulesResults(data);
        calculateOverallProgress(data);
      })
      .catch((error) => {
        console.error("Error fetching module results:", error);
        setError("Failed to fetch module results.");
      });
  }, [repositoryId, studentId]);

  const handleToggle = (moduleName) => {
    setActiveKey(activeKey === moduleName ? null : moduleName);
  };

  const getResultIcon = (isPassed) => {
    return isPassed ? (
      <FontAwesomeIcon icon={faCheckCircle} color="green" />
    ) : (
      <FontAwesomeIcon icon={faTimesCircle} color="red" />
    );
  };

  const calculateOverallProgress = (modules) => {
    // Aggregate all exercises across all modules
    const allExercises = Object.values(modules).reduce((acc, exercises) => {
      return acc.concat(Object.values(exercises));
    }, []);

    // Count all exercises that have passed
    const passedExercises = allExercises.filter(
      (ex) => ex.is_passed_test
    ).length;
    const totalExercises = allExercises.length;

    // Calculate the overall progress as a percentage
    const overallProgress =
      totalExercises > 0
        ? ((passedExercises / totalExercises) * 100).toFixed(2)
        : "0.00";
    setOverviewProgress(overallProgress);
  };

  return (
    <Container>
      <BackButton />
      <h1>Module Results for {studentId}</h1>
      <OverallProgress progress={overviewProgress} />
      {error && <Badge bg="danger">{error}</Badge>}
      <Accordion activeKey={activeKey}>
        {Object.entries(modulesResults).map(([moduleName, exercises], idx) => {
          const totalExercises = Object.values(exercises).length;
          const passedExercises = Object.values(exercises).filter(
            (ex) => ex.is_passed_test
          ).length;
          const progressPercent =
            totalExercises > 0 ? (passedExercises / totalExercises) * 100 : 0;

          return (
            <Card key={moduleName}>
              <Accordion.Item eventKey={moduleName}>
                <Accordion.Header onClick={() => handleToggle(moduleName)}>
                  <Row className="align-items-center">
                    <Col md={8}>
                      {moduleName.toUpperCase()} - Progress: {passedExercises}/
                      {totalExercises}
                    </Col>
                    <Col md={4}>
                      <ArrayProgress progressPercent={progressPercent} />
                    </Col>
                  </Row>
                  <FontAwesomeIcon
                    icon={
                      activeKey === moduleName ? faChevronUp : faChevronDown
                    }
                    className="ml-auto"
                  />
                </Accordion.Header>
                <Accordion.Body>
                  <ListGroup variant="flush">
                    {Object.entries(exercises).map(
                      ([exerciseName, exerciseDetails], index) => (
                        <ListGroup.Item key={index}>
                          {exerciseName}:{" "}
                          {getResultIcon(exerciseDetails.is_passed_test)}
                          <div>Score: {exerciseDetails.score}</div>
                          <div>
                            Logs:{" "}
                            {exerciseDetails.logs ? (
                              <pre style={{ whiteSpace: "pre-wrap" }}>
                                {exerciseDetails.logs}
                              </pre>
                            ) : (
                              "No logs available"
                            )}
                          </div>
                          <div>
                            Updated Time:{" "}
                            {exerciseDetails.updated_time_utc
                              ? formatDate(exerciseDetails.updated_time_utc)
                              : "Not updated"}
                          </div>
                        </ListGroup.Item>
                      )
                    )}
                  </ListGroup>
                </Accordion.Body>
              </Accordion.Item>
            </Card>
          );
        })}
      </Accordion>
    </Container>
  );
};

export default Student;
