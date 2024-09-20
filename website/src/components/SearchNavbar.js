// SearchNavbar.js
import React, { useState } from "react";
import { Form, FormControl } from "react-bootstrap";
import { useNavigate } from "react-router-dom";

const SearchNavbar = () => {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  const handleSearch = (e) => {
    e.preventDefault();
    if (query.trim() !== "") {
      navigate(`/search?query=${encodeURIComponent(query)}`);
    }
  };

  return (
    <Form className="d-flex" onSubmit={handleSearch}>
      <FormControl
        type="search"
        placeholder="Search"
        className="me-2"
        aria-label="Search"
        onChange={(e) => setQuery(e.target.value)}
        value={query}
      />
    </Form>
  );
};

export default SearchNavbar;
