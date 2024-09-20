import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Card, Button, Container, Alert, Row, Col } from "react-bootstrap";
import "styles/RepositoriesList.css";

const RepositoriesList = () => {
  const [repositories, setRepositories] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/repositories/repositories.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        const formattedData = Object.entries(data).map(([name, details]) => ({
          name,
          ...details,
        }));
        setRepositories(formattedData);
      })
      .catch((error) => {
        console.error("Error fetching Sessions Results:", error);
        setError("Failed to fetch repository data.");
      });
  }, []);

  return (
    <Container className="custom-repository-list">
      <h1 className="mb-4 mt-3">Sessions Results</h1>
      {error && <Alert variant="danger">{error}</Alert>}
      <Row xs={1} md={2} lg={3} className="g-4">
        {repositories.map((repo) => (
          <Col key={repo.name}>
            <Card className="repository-card">
              <Card.Body>
                <Card.Title>{repo.name}</Card.Title>
                <Card.Text>
                  Start Date: {repo.start_date}
                  <br />
                  End Date: {repo.end_date}
                  <br />
                  Number of Students: {repo.number_of_students}
                </Card.Text>
                <Card.Link
                  href={repo.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Repository URL
                </Card.Link>
                <Button
                  variant="primary"
                  as={Link}
                  to={`/students/${repo.name}`}
                  className="mt-3"
                >
                  View Students
                </Button>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </Container>
  );
};

export default RepositoriesList;
