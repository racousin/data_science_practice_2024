import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const CloudBasedSources = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Cloud-based Data Sources</h1>

      <section>
        <h2 id="object-storage">Object Storage Systems</h2>
        <p>
          Object storage is a data storage architecture that manages data as
          objects, as opposed to other storage architectures like file systems
          which manage data as a file hierarchy, and block storage which manages
          data as blocks within sectors and tracks.
        </p>

        <h3>Amazon S3</h3>
        <p>
          Amazon Simple Storage Service (S3) is a scalable, high-speed,
          web-based cloud storage service designed for online backup and
          archiving of data and application programs.
        </p>
        <CodeBlock
          language="python"
          code={`
import boto3
import pandas as pd

# Create an S3 client
s3 = boto3.client('s3', aws_access_key_id='YOUR_ACCESS_KEY', 
                  aws_secret_access_key='YOUR_SECRET_KEY')

# Upload a file to S3
s3.upload_file('local_file.csv', 'your-bucket-name', 'remote_file.csv')

# Download a file from S3
s3.download_file('your-bucket-name', 'remote_file.csv', 'local_file.csv')

# Read a CSV directly from S3 into a pandas DataFrame
df = pd.read_csv('s3://your-bucket-name/remote_file.csv')

# Write a DataFrame directly to S3
df.to_csv('s3://your-bucket-name/output_file.csv', index=False)
          `}
        />

        <h3>Google Cloud Storage</h3>
        <p>
          Google Cloud Storage is a RESTful online file storage web service for
          storing and accessing data on Google Cloud Platform infrastructure.
        </p>
        <CodeBlock
          language="python"
          code={`
from google.cloud import storage
import pandas as pd

# Create a client
client = storage.Client()

# Get the bucket
bucket = client.get_bucket('your-bucket-name')

# Upload a file
blob = bucket.blob('remote_file.csv')
blob.upload_from_filename('local_file.csv')

# Download a file
blob.download_to_filename('local_file.csv')

# Read a CSV directly from GCS into a pandas DataFrame
df = pd.read_csv('gs://your-bucket-name/remote_file.csv')

# Write a DataFrame directly to GCS
df.to_csv('gs://your-bucket-name/output_file.csv', index=False)
          `}
        />
      </section>

      <section>
        <h2 id="data-warehouses">Cloud Data Warehouses</h2>
        <p>
          Cloud data warehouses are fully managed, cloud-based data storage and
          analytics services that enable you to store large volumes of data and
          perform fast analytics.
        </p>

        <h3>Google BigQuery</h3>
        <p>
          BigQuery is Google's fully managed, serverless data warehouse that
          enables scalable analysis over petabytes of data.
        </p>
        <CodeBlock
          language="python"
          code={`
from google.cloud import bigquery
import pandas as pd

# Create a client
client = bigquery.Client()

# Perform a query
query = """
    SELECT name, COUNT(*) as count
    FROM bigquery-public-data.usa_names.usa_1910_2013
    WHERE state = 'TX'
    GROUP BY name
    ORDER BY count DESC
    LIMIT 10
"""
query_job = client.query(query)
results = query_job.result()

# Convert to DataFrame
df = pd.DataFrame(results.to_dataframe())
print(df)

# Load data from a pandas DataFrame into a new table
table_id = 'your_project.your_dataset.your_table'
job = client.load_table_from_dataframe(df, table_id)
job.result()  # Wait for the job to complete
          `}
        />

        <h3>Amazon Redshift</h3>
        <p>
          Amazon Redshift is a fully managed, petabyte-scale data warehouse
          service in the cloud.
        </p>
        <CodeBlock
          language="python"
          code={`
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Connect to Redshift
conn = psycopg2.connect(
    host='your_cluster.region.redshift.amazonaws.com',
    port=5439,
    dbname='your_database',
    user='your_username',
    password='your_password'
)

# Create a cursor
cur = conn.cursor()

# Execute a query
cur.execute("SELECT * FROM your_table LIMIT 10")
rows = cur.fetchall()

# Convert to DataFrame
df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
print(df)

# Use SQLAlchemy to write DataFrame to Redshift
engine = create_engine('postgresql://username:password@host:port/database')
df.to_sql('your_table', engine, if_exists='replace', index=False)

# Close the connection
conn.close()
          `}
        />
      </section>

      <section>
        <h2 id="python-sdks">Accessing Cloud Data with Python SDKs</h2>
        <p>
          Many cloud providers offer Python SDKs (Software Development Kits)
          that make it easier to interact with their services programmatically.
        </p>

        <h3>Azure Blob Storage</h3>
        <p>
          Azure Blob storage is Microsoft's object storage solution for the
          cloud.
        </p>
        <CodeBlock
          language="python"
          code={`
from azure.storage.blob import BlobServiceClient
import pandas as pd

# Create the BlobServiceClient object
connection_string = "your_connection_string"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get a container client
container_client = blob_service_client.get_container_client("your-container-name")

# Upload a file
with open("local_file.csv", "rb") as data:
    container_client.upload_blob(name="remote_file.csv", data=data)

# Download a file
with open("local_file.csv", "wb") as file:
    blob_client = container_client.get_blob_client("remote_file.csv")
    data = blob_client.download_blob()
    file.write(data.readall())

# Read a CSV directly from Azure Blob into a pandas DataFrame
df = pd.read_csv('https://your_account.blob.core.windows.net/your-container-name/remote_file.csv')
          `}
        />
      </section>

      <section>
        <h2>Best Practices for Working with Cloud-based Data Sources</h2>
        <ul>
          <li>
            <strong>Security:</strong> Always use secure methods to store and
            access your credentials. Never hardcode them in your scripts.
          </li>
          <li>
            <strong>Cost Management:</strong> Be aware of the costs associated
            with data storage and data transfer in cloud services.
          </li>
          <li>
            <strong>Data Governance:</strong> Implement proper data governance
            practices, including data cataloging and access control.
          </li>
          <li>
            <strong>Performance Optimization:</strong> Use appropriate file
            formats (like Parquet for analytics workloads) and partitioning
            strategies to optimize performance.
          </li>
          <li>
            <strong>Error Handling:</strong> Implement robust error handling and
            retry mechanisms when working with cloud services.
          </li>
          <li>
            <strong>Scalability:</strong> Design your data pipelines to be
            scalable, taking advantage of the elastic nature of cloud resources.
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default CloudBasedSources;
