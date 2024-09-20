import React, { useState } from "react";
import { Container, Row, Col, Card, Button, Form } from "react-bootstrap";
import "styles/Resources.css";

const Resources = () => {
  const [selectedTag, setSelectedTag] = useState("All");

  const resources = [
    {
      id: 0,
      title: "Sorbonne Université",
      link: "https://m2stat.sorbonne-universite.fr/",
      tags: ["Education", "Research", "AI"],
      logo_url: "/assets/resources/sorbonne.png",
    },

    {
      id: 1,
      title: "Git",
      link: "https://git-scm.com",
      tags: ["Version Control"],
      logo_url: "/assets/resources/git.jpg",
    },
    {
      id: 2,
      title: "Docker",
      link: "https://docker.com",
      tags: ["DevOps", "Containers"],
      logo_url: "/assets/resources/docker.svg",
    },
    {
      id: 3,
      title: "Kubernetes",
      link: "https://kubernetes.io",
      tags: ["DevOps", "Containers"],
      logo_url: "/assets/resources/k8s.png",
    },
    {
      id: 4,
      title: "PyTorch",
      link: "https://pytorch.org",
      tags: ["Deep Learning"],
      logo_url: "/assets/resources/pytorch.png",
    },
    {
      id: 5,
      title: "TensorFlow",
      link: "https://tensorflow.org",
      tags: ["Deep Learning"],
      logo_url: "/assets/resources/tensorflow.png",
    },
    {
      id: 6,
      title: "Scikit-Learn",
      link: "https://scikit-learn.org",
      tags: ["Machine Learning"],
      logo_url: "/assets/resources/scikitlearn.png",
    },
    {
      id: 7,
      title: "Kaggle",
      link: "https://kaggle.com",
      tags: ["Competitions", "Datasets"],
      logo_url: "/assets/resources/Kaggle.png",
    },
    {
      id: 8,
      title: "Hugging Face",
      link: "https://huggingface.co",
      tags: ["NLP", "Models", "Datasets"],
      logo_url: "/assets/resources/HuggingFace.png",
    },
    {
      id: 9,
      title: "Google Cloud Platform",
      link: "https://cloud.google.com",
      tags: ["Cloud", "AI"],
      logo_url: "/assets/resources/gcp.png",
    },
    {
      id: 10,
      title: "AWS",
      link: "https://aws.amazon.com",
      tags: ["Cloud"],
      logo_url: "/assets/resources/aws.png",
    },
    {
      id: 11,
      title: "Azure",
      link: "https://azure.microsoft.com",
      tags: ["Cloud"],
      logo_url: "/assets/resources/Azure.png",
    },
    {
      id: 12,
      title: "RL Arena",
      link: "https://rlarena.com",
      tags: ["Reinforcement Learning", "Competitions"],
      logo_url: "/assets/resources/rlarena.png",
    },
    {
      id: 13,
      title: "Codestral",
      link: "https://mistral.ai/news/codestral/",
      tags: ["Development", "Tools"],
      logo_url: "/assets/resources/mistralai.jpeg",
    },
    {
      id: 14,
      title: "OpenAI",
      link: "https://openai.com",
      tags: ["AI", "Research", "Deep Learning"],
      logo_url: "/assets/resources/openai.png",
    },
    {
      id: 15,
      title: "DeepMind",
      link: "https://deepmind.com",
      tags: ["AI", "Research", "Deep Learning"],
      logo_url: "/assets/resources/DeepMind.png",
    },
    {
      id: 16,
      title: "GitHub",
      link: "https://github.com",
      tags: ["Development", "Tools", "Version Control"],
      logo_url: "/assets/resources/github.png",
    },
    {
      id: 17,
      title: "PostgreSQL",
      link: "https://www.postgresql.org",
      tags: ["Databases", "Tools"],
      logo_url: "/assets/resources/postgresql.png",
    },
    {
      id: 18,
      title: "Hacker News",
      link: "https://news.ycombinator.com",
      tags: ["News", "Technology", "Community"],
      logo_url: "/assets/resources/hackernews.png",
    },
    {
      id: 20,
      title: "Gymnasium",
      link: "https://gymnasium.tech",
      tags: ["Education", "Training", "Technology"],
      logo_url: "/assets/resources/Gymnasium.png",
    },
    {
      id: 21,
      title: "Stanford University",
      link: "https://online.stanford.edu/courses/topics/artificial-intelligence",
      tags: ["Education", "AI", "Research"],
      logo_url: "/assets/resources/stanford.png",
    },
    {
      id: 22,
      title: "University of Oxford",
      link: "https://www.cs.ox.ac.uk/research/ai/",
      tags: ["Education", "AI", "Research"],
      logo_url: "/assets/resources/oxford.jpeg",
    },
    {
      id: 23,
      title: "UC Berkeley",
      link: "https://ai.berkeley.edu",
      tags: ["Education", "AI", "Research"],
      logo_url: "/assets/resources/Berkeley.png",
    },
    {
      id: 24,
      title: "Coursera",
      link: "https://www.coursera.org",
      tags: ["Online Courses", "Education", "AI"],
      logo_url: "/assets/resources/coursera.png",
    },
    {
      id: 25,
      title: "MIT OpenCourseWare",
      link: "https://ocw.mit.edu",
      tags: ["Online Courses", "Education", "Technology"],
      logo_url: "/assets/resources/mitocw.png",
    },
  ];

  const tags = [
    "All",
    "Version Control",
    "DevOps",
    "Containers",
    "Deep Learning",
    "Machine Learning",
    "Competitions",
    "Datasets",
    "NLP",
    "Models",
    "Cloud",
    "Reinforcement Learning",
    "AI",
    "Research",
    "Development",
    "Tools",
    "Web Development",
    "Databases",
    "News",
    "Technology",
    "Community",
    "Education",
  ];

  const handleSelectTag = (e) => {
    setSelectedTag(e.target.value);
  };

  const filteredResources =
    selectedTag === "All"
      ? resources
      : resources.filter((resource) => resource.tags.includes(selectedTag));

  return (
    <Container className="resources-container">
      <h1 className="my-4">Useful Resources</h1>
      <Form>
        <Form.Group controlId="tagSelect">
          <Form.Label>Select a tag to filter resources:</Form.Label>
          <Form.Control
            as="select"
            value={selectedTag}
            onChange={handleSelectTag}
          >
            {tags.map((tag) => (
              <option key={tag}>{tag}</option>
            ))}
          </Form.Control>
        </Form.Group>
      </Form>
      <Row>
        {filteredResources.map((resource) => (
          <Col key={resource.id} md={4} className="mb-4">
            <Card className="resources-card">
              <Card.Img
                variant="top"
                src={resource.logo_url || "path/to/generic_logo.png"}
                className="resources-logo-image"
              />
              <Card.Body>
                <Card.Title className="resources-title">
                  {resource.title}
                </Card.Title>
                <Button
                  variant="primary"
                  href={resource.link}
                  className="resources-button"
                >
                  Learn More
                </Button>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </Container>
  );
};

export default Resources;
