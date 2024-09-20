import React from "react";
import { Container } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DockerComposeServices = () => {
  return (
    <Container>
      <h1>Docker Compose and Services</h1>

      <section id="docker-compose-intro">
        <h2>Introduction to Docker Compose</h2>
        <p>
          Docker Compose is a tool for defining and running multi-container
          Docker applications. With Compose, you use a YAML file to configure
          your application's services, networks, and volumes.
        </p>
        <p>Key benefits of using Docker Compose include:</p>
        <ul>
          <li>Single command to spin up entire application stack</li>
          <li>Easy configuration management</li>
          <li>Environment variable management</li>
          <li>Service dependencies and scaling</li>
        </ul>
      </section>

      <section id="compose-file-structure">
        <h2>Docker Compose File Structure</h2>
        <p>
          A Docker Compose file (typically named docker-compose.yml) defines the
          services, networks, and volumes for a Docker application. Here's a
          basic structure:
        </p>
        <CodeBlock
          code={`
version: '3'
services:
  web:
    build: .
    ports:
      - "5000:5000"
  redis:
    image: "redis:alpine"

volumes:
  logvolume01: {}

networks:
  frontend:
  backend:
          `}
          language="yaml"
        />
        <p>
          This example defines two services (web and redis), a volume, and two
          networks.
        </p>
      </section>

      <section id="multi-container-apps">
        <h2>Creating Multi-Container Applications</h2>
        <p>
          Let's create a more complex example with a web application, a
          database, and a redis cache:
        </p>
        <CodeBlock
          code={`
version: '3'
services:
  web:
    build: ./web
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/dbname
      - REDIS_URL=redis://redis:6379
  db:
    image: postgres:12
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=dbname
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
  redis:
    image: "redis:alpine"

volumes:
  postgres_data:
          `}
          language="yaml"
        />
        <p>To run this multi-container application:</p>
        <CodeBlock
          code={`
# Start the application
docker-compose up -d

# Stop the application
docker-compose down

# View logs
docker-compose logs

# Scale a service
docker-compose up -d --scale web=3
          `}
          language="bash"
        />
      </section>

      <section id="docker-services">
        <h2>Docker Services and Swarm Mode</h2>
        <p>
          Docker Swarm mode allows you to manage a cluster of Docker engines.
          Services are the core concept in Swarm mode, representing a group of
          container instances running the same image.
        </p>
        <p>To use Swarm mode:</p>
        <CodeBlock
          code={`
# Initialize a swarm
docker swarm init

# Create a service
docker service create --name my_web nginx

# List services
docker service ls

# Scale a service
docker service scale my_web=3

# Update a service
docker service update --image nginx:1.14 my_web

# Remove a service
docker service rm my_web
          `}
          language="bash"
        />
        <p>Docker Compose can also be used with Swarm mode to deploy stacks:</p>
        <CodeBlock
          code={`
# Deploy a stack
docker stack deploy -c docker-compose.yml my_stack

# List stacks
docker stack ls

# Remove a stack
docker stack rm my_stack
          `}
          language="bash"
        />
      </section>

      <section id="networking-compose">
        <h2>Networking in Docker Compose</h2>
        <p>
          Docker Compose sets up a single network for your app by default, but
          you can also specify custom networks:
        </p>
        <CodeBlock
          code={`
version: '3'
services:
  web:
    build: .
    networks:
      - frontend
      - backend
  db:
    image: postgres
    networks:
      - backend

networks:
  frontend:
  backend:
          `}
          language="yaml"
        />
        <p>
          This setup creates two networks and connects the web service to both,
          while the db service is only connected to the backend network.
        </p>
      </section>

      <section id="conclusion">
        <h2>Conclusion</h2>
        <p>
          Docker Compose and Services provide powerful tools for defining,
          running, and scaling multi-container applications. They simplify the
          process of managing complex applications and are essential for both
          development and production environments.
        </p>
      </section>
    </Container>
  );
};

export default DockerComposeServices;
