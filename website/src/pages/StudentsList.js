import React, { useState, useEffect, useMemo } from "react";
import { Link, useParams } from "react-router-dom";
import { Table, Container, Alert, Form, Button } from "react-bootstrap";
import "styles/StudentsList.css"; // Ensure to create this CSS for additional styling
import BackButton from "components/BackButton";
import ArrayProgress from "components/ArrayProgress";

const StudentsList = () => {
  const { repositoryId } = useParams();
  const [students, setStudents] = useState([]);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const [sortConfig, setSortConfig] = useState({ key: null, direction: null });

  useEffect(() => {
    fetch(`/repositories/${repositoryId}/students/config/students.json`)
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
        setStudents(formattedData);
      })
      .catch((error) => {
        console.error("Error fetching student list:", error);
        setError("Failed to fetch student data.");
      });
  }, [repositoryId]);

  const sortedStudents = useMemo(() => {
    let sortableItems = [...students];
    if (sortConfig !== null) {
      sortableItems.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === "ascending" ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === "ascending" ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableItems;
  }, [students, sortConfig]);

  const filteredStudents = useMemo(() => {
    return sortedStudents.filter((student) =>
      student.name.toLowerCase().includes(filter.toLowerCase())
    );
  }, [sortedStudents, filter]);

  const requestSort = (key) => {
    let direction = "ascending";
    if (sortConfig.key === key && sortConfig.direction === "ascending") {
      direction = "descending";
    }
    setSortConfig({ key, direction });
  };

  return (
    <Container className="students-list-container">
      <BackButton />
      <h1 className="mb-4">Student List for {repositoryId}</h1>
      {error && <Alert variant="danger">{error}</Alert>}
      <Form className="d-flex mb-3">
        <Form.Control
          type="text"
          placeholder="Filter by name..."
          onChange={(e) => setFilter(e.target.value)}
          value={filter}
        />
      </Form>
      <Table striped bordered hover responsive>
        <thead>
          <tr>
            <th onClick={() => requestSort("name")}>Name</th>
            <th onClick={() => requestSort("progress_percentage")}>Progress</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {filteredStudents.map((student, index) => (
            <tr key={index}>
              <td>{student.name}</td>
              <td>
                <ArrayProgress
                  progressPercent={student.progress_percentage * 100}
                />
              </td>
              <td>
                <Link
                  to={`/student/${repositoryId}/${student.name}`}
                  className="btn btn-primary"
                >
                  View Details
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </Container>
  );
};

export default StudentsList;
