import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const FileBased = () => {
  return (
    <Container fluid>
      <h1 className="my-4">File-based Data Sources</h1>

      <section>
        <h2 id="file-formats">CSV, JSON, and XML Files</h2>

        <h3>CSV (Comma-Separated Values)</h3>
        <p>
          CSV files are simple, tabular data files where each row represents a
          record and columns are separated by commas.
        </p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Reading a CSV file
df = pd.read_csv('data.csv')
print(df.head())

# Writing to a CSV file
df.to_csv('output.csv', index=False)
          `}
        />

        <h3>JSON (JavaScript Object Notation)</h3>
        <p>
          JSON is a lightweight data interchange format that is easy for humans
          to read and write and easy for machines to parse and generate.
        </p>
        <CodeBlock
          language="python"
          code={`
import json
import pandas as pd

# Reading a JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame if it's a list of dictionaries
df = pd.DataFrame(data)
print(df.head())

# Writing to a JSON file
df.to_json('output.json', orient='records')
          `}
        />

        <h3>XML (eXtensible Markup Language)</h3>
        <p>
          XML is a markup language that defines a set of rules for encoding
          documents in a format that is both human-readable and
          machine-readable.
        </p>
        <CodeBlock
          language="python"
          code={`
import xml.etree.ElementTree as ET
import pandas as pd

# Reading an XML file
tree = ET.parse('data.xml')
root = tree.getroot()

# Convert to DataFrame (example for simple structure)
data = []
for child in root:
    data.append({elem.tag: elem.text for elem in child})
df = pd.DataFrame(data)
print(df.head())

# Writing to an XML file
def dataframe_to_xml(df, filename):
    root = ET.Element('root')
    for _, row in df.iterrows():
        child = ET.SubElement(root, 'item')
        for col, value in row.items():
            ET.SubElement(child, col).text = str(value)
    tree = ET.ElementTree(root)
    tree.write(filename)

dataframe_to_xml(df, 'output.xml')
          `}
        />
      </section>

      <section>
        <h2 id="pandas-io">
          Reading and Writing Various File Formats using Pandas
        </h2>
        <p>
          Pandas provides a unified interface for reading and writing various
          file formats.
        </p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Excel files
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df_excel.to_excel('output.xlsx', index=False)

# Parquet files (columnar storage format)
df_parquet = pd.read_parquet('data.parquet')
df_parquet.to_parquet('output.parquet')

# HDF5 files (hierarchical data format)
df_hdf = pd.read_hdf('data.h5', key='df')
df_hdf.to_hdf('output.h5', key='df', mode='w')

# Feather files (fast, lightweight binary format)
df_feather = pd.read_feather('data.feather')
df_feather.to_feather('output.feather')

# Pickle files (Python object serialization)
df_pickle = pd.read_pickle('data.pkl')
df_pickle.to_pickle('output.pkl')
          `}
        />
      </section>

      <section>
        <h2 id="large-files">Handling Large Files: Chunking and Iterators</h2>
        <p>
          When dealing with large files that don't fit into memory, we can use
          chunking and iterators to process the data in smaller pieces.
        </p>

        <h3>Chunking</h3>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Reading a large CSV file in chunks
chunk_size = 10000  # Adjust based on your system's memory
chunks = pd.read_csv('large_file.csv', chunksize=chunk_size)

# Process each chunk
for chunk in chunks:
    # Perform operations on the chunk
    processed_chunk = chunk.some_operation()
    
    # Optionally, save processed chunk
    processed_chunk.to_csv('output.csv', mode='a', header=False, index=False)
          `}
        />

        <h3>Iterators</h3>
        <CodeBlock
          language="python"
          code={`
import pandas as pd

# Create an iterator for a large CSV file
df_iterator = pd.read_csv('large_file.csv', iterator=True)

# Read and process 1000 rows at a time
while True:
    try:
        chunk = df_iterator.get_chunk(1000)
        # Process the chunk
        processed_chunk = chunk.some_operation()
        # Optionally, save processed chunk
        processed_chunk.to_csv('output.csv', mode='a', header=False, index=False)
    except StopIteration:
        break  # End of file reached
          `}
        />

        <h3>Memory-Efficient Operations</h3>
        <p>
          When working with large files, it's important to use memory-efficient
          operations:
        </p>
        <ul>
          <li>Use generators instead of lists where possible</li>
          <li>Avoid creating unnecessary copies of data</li>
          <li>Use inplace=True for pandas operations when applicable</li>
          <li>
            Consider using libraries like Dask or Vaex for out-of-core
            computations
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default FileBased;
