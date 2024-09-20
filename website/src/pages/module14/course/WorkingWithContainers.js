import React from "react";
import { Container, Table } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const WorkingWithContainers = () => {
  return (
    <Container>
      <h1>Working with Docker Containers</h1>

      <section id="running-containers">
        <h2>Running Containers</h2>
        <p>
          Running containers is the core functionality of Docker. Here are some
          common ways to run containers:
        </p>
        <CodeBlock
          code={`
# Run a container in detached mode
docker run -d nginx

# Run a container with a custom name
docker run --name my-nginx -d nginx

# Run a container and publish a port
docker run -d -p 8080:80 nginx

# Run a container with environment variables
docker run -d -e MY_VAR=my_value nginx

# Run a container with a volume
docker run -d -v /host/path:/container/path nginx
          `}
          language="bash"
        />
      </section>

      <section id="managing-containers">
        <h2>Managing Containers</h2>
        <p>
          Docker provides various commands to manage the lifecycle of
          containers:
        </p>
        <CodeBlock
          code={`
# List running containers
docker ps

# List all containers (including stopped ones)
docker ps -a

# Stop a running container
docker stop <container_id>

# Start a stopped container
docker start <container_id>

# Restart a container
docker restart <container_id>

# Remove a container
docker rm <container_id>

# Remove all stopped containers
docker container prune
          `}
          language="bash"
        />
      </section>

      <section id="container-lifecycle">
        <h2>Container Lifecycle</h2>
        <p>
          Understanding the container lifecycle is crucial for effective
          container management:
        </p>
        <Table striped bordered hover>
          <thead>
            <tr>
              <th>State</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Created</td>
              <td>Container is created but not started</td>
            </tr>
            <tr>
              <td>Running</td>
              <td>Container is running with all its processes</td>
            </tr>
            <tr>
              <td>Paused</td>
              <td>Container processes are paused</td>
            </tr>
            <tr>
              <td>Stopped</td>
              <td>Container processes are stopped</td>
            </tr>
            <tr>
              <td>Deleted</td>
              <td>Container is deleted and no longer exists</td>
            </tr>
          </tbody>
        </Table>
      </section>

      <section id="container-networking">
        <h2>Container Networking</h2>
        <p>
          Docker provides powerful networking capabilities to connect containers
          to each other and to the outside world:
        </p>
        <CodeBlock
          code={`
# Create a network
docker network create my-network

# Run a container and connect it to a network
docker run -d --name my-nginx --network my-network nginx

# Inspect a network
docker network inspect my-network

# Connect a running container to a network
docker network connect my-network <container_id>

# Disconnect a container from a network
docker network disconnect my-network <container_id>
          `}
          language="bash"
        />
      </section>

      <section id="data-management">
        <h2>Data Management and Volumes</h2>
        <p>
          Managing data in Docker containers is crucial for persistence and
          sharing data between containers:
        </p>
        <CodeBlock
          code={`
# Create a volume
docker volume create my-volume

# Run a container with a volume
docker run -d -v my-volume:/app/data nginx

# Inspect a volume
docker volume inspect my-volume

# List volumes
docker volume ls

# Remove a volume
docker volume rm my-volume

# Remove all unused volumes
docker volume prune
          `}
          language="bash"
        />
        <p>
          Volumes are the preferred way to persist data in Docker containers.
          They are managed by Docker and are isolated from the core
          functionality of the host machine.
        </p>
      </section>

      <section id="container-logs">
        <h2>Container Logs and Debugging</h2>
        <p>
          Accessing logs and debugging containers is essential for
          troubleshooting:
        </p>
        <CodeBlock
          code={`
# View container logs
docker logs <container_id>

# Follow container logs
docker logs -f <container_id>

# View last n lines of container logs
docker logs --tail n <container_id>

# Execute a command in a running container
docker exec -it <container_id> /bin/bash

# View container resource usage statistics
docker stats <container_id>
          `}
          language="bash"
        />
      </section>

      <section id="conclusion">
        <h2>Conclusion</h2>
        <p>
          Working with Docker containers involves understanding how to run,
          manage, network, and debug them. These skills form the foundation of
          using Docker effectively in various scenarios, from development to
          production environments.
        </p>
      </section>
    </Container>
  );
};

export default WorkingWithContainers;
