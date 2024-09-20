import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const RealTimeStreams = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Real-time Data Streams</h1>

      <section>
        <h2 id="intro-streaming">Introduction to Streaming Data</h2>
        <p>
          Streaming data refers to data that is generated continuously, often in
          high volumes and at high velocity. Examples include:
        </p>
        <ul>
          <li>Social media feeds</li>
          <li>Financial market data</li>
          <li>IoT sensor readings</li>
          <li>Log files from web servers</li>
        </ul>
        <p>
          Processing streaming data requires different approaches compared to
          batch processing, as data arrives in real-time and often needs to be
          processed immediately.
        </p>
      </section>

      <section>
        <h2 id="kafka">Working with Apache Kafka</h2>
        <p>
          Apache Kafka is a distributed streaming platform that allows you to
          publish and subscribe to streams of records.
        </p>

        <h3>Key Concepts in Kafka</h3>
        <ul>
          <li>
            <strong>Topics:</strong> Categories or feed names to which records
            are published
          </li>
          <li>
            <strong>Producers:</strong> Clients that publish data to Kafka
            topics
          </li>
          <li>
            <strong>Consumers:</strong> Clients that subscribe to topics and
            process the published data
          </li>
          <li>
            <strong>Brokers:</strong> Servers that store the published data
          </li>
        </ul>

        <h3>Using Kafka with Python</h3>
        <CodeBlock
          language="python"
          code={`
from kafka import KafkaProducer, KafkaConsumer
import json

# Producer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Produce a message
producer.send('my-topic', {'key': 'value'})

# Consumer
consumer = KafkaConsumer('my-topic',
                         bootstrap_servers=['localhost:9092'],
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))

# Consume messages
for message in consumer:
    print(message.value)
          `}
        />

        <h3>Streaming Analytics with Kafka Streams</h3>
        <p>
          Kafka Streams is a client library for building applications and
          microservices that process and analyze data stored in Kafka.
        </p>
        <CodeBlock
          language="python"
          code={`
from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.streams import StreamsBuilder, KafkaStreams
import json

# Define a simple streaming application
builder = StreamsBuilder()
input_topic = builder.stream('input-topic')
output_topic = input_topic.map(lambda k, v: (k, v.upper()))
output_topic.to('output-topic')

# Build and start the streams
streams = KafkaStreams(builder, {'bootstrap.servers': 'localhost:9092'})
streams.start()

# Remember to close the streams when done
# streams.close()
          `}
        />
      </section>

      <section>
        <h2 id="processing-streams">Processing Streams with Python</h2>
        <p>
          Python offers several libraries for processing streaming data beyond
          Kafka.
        </p>

        <h3>Using Apache Flink with PyFlink</h3>
        <p>
          Apache Flink is a framework and distributed processing engine for
          stateful computations over unbounded and bounded data streams.
        </p>
        <CodeBlock
          language="python"
          code={`
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, DataTypes
from pyflink.table.expressions import col

env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(env)

# Define a source table
t_env.execute_sql("""
    CREATE TABLE source_table (
        id BIGINT,
        data STRING
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'source_topic',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
""")

# Define a sink table
t_env.execute_sql("""
    CREATE TABLE sink_table (
        id BIGINT,
        data STRING
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'sink_topic',
        'properties.bootstrap.servers' = 'localhost:9092',
        'format' = 'json'
    )
""")

# Define a simple transformation
source_table = t_env.from_path('source_table')
result_table = source_table.select(col('id'), col('data').upper_case())

# Execute the job
result_table.execute_insert('sink_table').wait()
          `}
        />

        <h3>Real-time Processing with Python's asyncio</h3>
        <p>
          For smaller-scale stream processing, Python's built-in asyncio library
          can be very useful.
        </p>
        <CodeBlock
          language="python"
          code={`
import asyncio
import aiohttp
import json

async def fetch_data(session, url):
    async with session.get(url) as response:
        return await response.json()

async def process_stream():
    async with aiohttp.ClientSession() as session:
        while True:
            data = await fetch_data(session, 'http://example.com/stream')
            # Process the data
            processed_data = json.dumps(data).upper()
            print(processed_data)
            await asyncio.sleep(1)  # Wait for 1 second before next request

asyncio.run(process_stream())
          `}
        />
      </section>

      <section>
        <h2>Best Practices for Stream Processing</h2>
        <ul>
          <li>
            <strong>Fault Tolerance:</strong> Implement mechanisms to handle
            failures and ensure data isn't lost.
          </li>
          <li>
            <strong>Scalability:</strong> Design your system to handle
            increasing data volumes by adding more processing units.
          </li>
          <li>
            <strong>State Management:</strong> Carefully manage state in your
            streaming applications, especially for operations like windowing and
            aggregations.
          </li>
          <li>
            <strong>Monitoring and Alerting:</strong> Implement robust
            monitoring to quickly detect and respond to issues in your streaming
            pipeline.
          </li>
          <li>
            <strong>Data Quality:</strong> Implement data quality checks as part
            of your stream processing to catch and handle bad data early.
          </li>
          <li>
            <strong>Exactly-Once Processing:</strong> When required, ensure your
            system processes each record exactly once, even in the face of
            failures.
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default RealTimeStreams;
