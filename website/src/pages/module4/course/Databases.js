import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Databases = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Database Systems</h1>

      <section>
        <h2 id="relational">Relational Databases (SQL)</h2>
        <p>
          Relational databases use structured query language (SQL) and store
          data in tables with predefined schemas. They are ideal for structured
          data with complex relationships.
        </p>

        <h3>Connecting to Databases (SQLAlchemy)</h3>
        <p>
          SQLAlchemy is a popular Python SQL toolkit and Object-Relational
          Mapping (ORM) library.
        </p>
        <CodeBlock
          language="python"
          code={`
from sqlalchemy import create_engine
import pandas as pd

# Create a connection to the database
engine = create_engine('postgresql://username:password@host:port/database')

# Read data from a table into a pandas DataFrame
df = pd.read_sql_table('table_name', engine)

# Or execute a custom query
df = pd.read_sql_query('SELECT * FROM table_name WHERE condition', engine)

# Write data to a table
df.to_sql('table_name', engine, if_exists='replace', index=False)
          `}
        />

        <h3>Writing Efficient SQL Queries</h3>
        <p>Efficient SQL queries are crucial for good database performance.</p>
        <CodeBlock
          language="sql"
          code={`
-- Use appropriate indexing
CREATE INDEX idx_column ON table_name(column_name);

-- Avoid using SELECT *
SELECT specific_column1, specific_column2 FROM table_name;

-- Use JOINs instead of subqueries when possible
SELECT a.column1, b.column2
FROM table1 a
JOIN table2 b ON a.id = b.id;

-- Use EXPLAIN to analyze query performance
EXPLAIN SELECT * FROM table_name WHERE condition;
          `}
        />

        <h3>Fetching Data into Pandas DataFrames</h3>
        <p>Pandas provides convenient methods to work with SQL databases.</p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://username:password@host:port/database')

# Read an entire table
df = pd.read_sql_table('table_name', engine)

# Execute a custom query
query = """
SELECT column1, column2
FROM table_name
WHERE condition
GROUP BY column1
HAVING aggregate_condition
ORDER BY column2
LIMIT 1000
"""
df = pd.read_sql_query(query, engine)

# Iterate over large result sets
for chunk in pd.read_sql_query(query, engine, chunksize=10000):
    process_data(chunk)
          `}
        />
      </section>

      <section>
        <h2 id="nosql">NoSQL Databases</h2>
        <p>
          NoSQL databases provide flexible schemas for unstructured and
          semi-structured data. They are often used for large-scale, distributed
          data storage and retrieval.
        </p>

        <h3>Document Stores (MongoDB)</h3>
        <p>
          MongoDB is a popular document-oriented database that stores data in
          flexible, JSON-like documents.
        </p>
        <CodeBlock
          language="python"
          code={`
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['database_name']
collection = db['collection_name']

# Insert data
data = {"name": "John Doe", "age": 30, "city": "New York"}
collection.insert_one(data)

# Query data
result = collection.find({"age": {"$gt": 25}})
df = pd.DataFrame(list(result))

# Update data
collection.update_one({"name": "John Doe"}, {"$set": {"age": 31}})

# Delete data
collection.delete_one({"name": "John Doe"})
          `}
        />

        <h3>Key-Value Stores (Redis)</h3>
        <p>
          Redis is an in-memory data structure store, used as a database, cache,
          and message broker.
        </p>
        <CodeBlock
          language="python"
          code={`
import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Set a key-value pair
r.set('user:1', json.dumps({"name": "John Doe", "age": 30}))

# Get a value
user = json.loads(r.get('user:1'))

# Set with expiration
r.setex('session:123', 3600, 'active')  # expires in 1 hour

# Increment a value
r.incr('visitors')

# Delete a key
r.delete('user:1')
          `}
        />

        <h3>Graph Databases (Neo4j)</h3>
        <p>
          Neo4j is a graph database management system that stores and processes
          data in nodes and relationships.
        </p>
        <CodeBlock
          language="python"
          code={`
from neo4j import GraphDatabase

# Connect to Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def add_person(tx, name):
    tx.run("CREATE (a:Person {name: $name})", name=name)

def find_person(tx, name):
    result = tx.run("MATCH (a:Person {name: $name}) RETURN a.name AS name", name=name)
    return [record["name"] for record in result]

with driver.session() as session:
    # Add a person
    session.write_transaction(add_person, "Alice")
    
    # Find a person
    names = session.read_transaction(find_person, "Alice")
    for name in names:
        print(name)

driver.close()
          `}
        />
      </section>
    </Container>
  );
};

export default Databases;
