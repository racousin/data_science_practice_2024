// SearchResultsPage.js
import React, { useEffect, useState } from "react";
import { useLocation, Link } from "react-router-dom";
import Fuse from "fuse.js";
import { Card, ListGroup } from "react-bootstrap"; // Using Bootstrap components for styling

const SearchResultsPage = () => {
  const [results, setResults] = useState([]);
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const query = queryParams.get("query");

  const documents = [
    {
      title: "Home Page",
      content: "Welcome to the Data Science Practice Portal.",
      path: "/",
    },
    {
      title: "Teaching",
      content: "Explore teaching materials.",
      path: "/teaching",
    },
    {
      title: "Sessions Results",
      content: "View results of past sessions.",
      path: "/repositories",
    },
  ];

  const options = {
    includeScore: true,
    keys: ["title", "content"],
  };

  useEffect(() => {
    const fuse = new Fuse(documents, options);
    if (query) {
      const searchResults = fuse.search(query);
      setResults(searchResults.map((result) => result.item));
    }
  }, [query]);

  return (
    <div className="container mt-4">
      <h1>Search Results for: "{query}"</h1>
      {results.length > 0 ? (
        <ListGroup>
          {results.map((item, index) => (
            <ListGroup.Item
              key={index}
              action
              as={Link}
              to={item.path}
              className="p-3"
            >
              <Card>
                <Card.Body>
                  <Card.Title>{item.title}</Card.Title>
                  <Card.Text>{item.content}</Card.Text>
                </Card.Body>
              </Card>
            </ListGroup.Item>
          ))}
        </ListGroup>
      ) : (
        <p>No results found.</p>
      )}
    </div>
  );
};

export default SearchResultsPage;
