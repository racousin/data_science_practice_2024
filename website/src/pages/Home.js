import React from "react";
import { Container, Row, Col, Card, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import StartLearningButton from "components/StartLearningButton";

const Home = () => {
  return (
    <Container fluid className="p-0">
      <Row
        className="align-items-center"
        style={{
          height: "65vh",
        }}
      >
        <Col className="text-center">
          <h1 className="text-gray display-3 font-weight-bold">
            Welcome to the Data Science Practice Portal
          </h1>
          <StartLearningButton />
        </Col>
      </Row>
      <Row className="m-3 text-center">
        <Col md={4} className="mb-4">
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Teaching</Card.Title>
              <Card.Text>
                Explore a variety of modules designed to enhance your skills and
                knowledge in data science and related fields.
              </Card.Text>
              <Button variant="outline-primary" as={Link} to="/teaching">
                Learn More
              </Button>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4} className="mb-4">
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Sessions Results</Card.Title>
              <Card.Text>
                View your progress and results from past sessions. Track your
                achievements and identify areas for improvement.
              </Card.Text>
              <Button variant="outline-success" as={Link} to="/repositories">
                View Results
              </Button>
            </Card.Body>
          </Card>
        </Col>
        <Col md={4} className="mb-4">
          <Card className="h-100">
            <Card.Body>
              <Card.Title>Resources</Card.Title>
              <Card.Text>
                Access a wealth of resources such as tutorials, articles, and
                tools to support your ongoing learning and projects.
              </Card.Text>
              <Button variant="outline-info" as={Link} to="/resources">
                Explore Resources
              </Button>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Home;
