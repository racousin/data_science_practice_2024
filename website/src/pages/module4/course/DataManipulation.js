import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const DataManipulation = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Data Manipulation with Pandas</h1>

      <section>
        <h2 id="dataframe-series">DataFrame and Series Objects</h2>
        <p>
          Pandas provides two primary data structures: DataFrame and Series. A
          DataFrame is a 2-dimensional labeled data structure with columns of
          potentially different types. A Series is a 1-dimensional labeled array
          that can hold data of any type.
        </p>
        <CodeBlock
          language="python"
          code={`
import pandas as pd
import numpy as np

# Creating a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

# Creating a DataFrame
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

# DataFrame from dict
df2 = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20130102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'
})
print(df2)
          `}
        />
      </section>

      <section>
        <h2 id="loading-cleaning">Data Loading, Cleaning, and Preprocessing</h2>
        <p>
          Pandas provides various functions to load data from different sources
          and clean it.
        </p>
        <CodeBlock
          language="python"
          code={`
# Loading data
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')
df = pd.read_sql('SELECT * FROM table', connection)

# Basic info about the DataFrame
print(df.info())
print(df.describe())

# Checking for missing values
print(df.isnull().sum())

# Dropping missing values
df_cleaned = df.dropna()

# Filling missing values
df_filled = df.fillna(value={'numeric_col': 0, 'string_col': 'Unknown'})

# Removing duplicates
df_unique = df.drop_duplicates()

# Data type conversion
df['date_col'] = pd.to_datetime(df['date_col'])
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')

# Renaming columns
df = df.rename(columns={'old_name': 'new_name'})
          `}
        />
      </section>

      <section>
        <h2 id="merging-joining">
          Merging, Joining, and Concatenating Datasets
        </h2>
        <p>Pandas offers various ways to combine datasets.</p>
        <CodeBlock
          language="python"
          code={`
# Concatenating DataFrames
df_concat = pd.concat([df1, df2])

# Merging DataFrames
df_merged = pd.merge(df1, df2, on='key_column')

# Joining DataFrames
df_joined = df1.join(df2, on='key_column', how='left')

# Combining DataFrames with different columns
df_combined = df1.combine_first(df2)
          `}
        />
      </section>

      <section>
        <h2 id="grouping-aggregation">Grouping and Aggregation Operations</h2>
        <p>Grouping and aggregation are powerful features for data analysis.</p>
        <CodeBlock
          language="python"
          code={`
# Grouping and aggregation
grouped = df.groupby('category')
print(grouped['value'].mean())

# Multiple aggregations
agg_df = df.groupby('category').agg({
    'numeric_col': ['mean', 'median', 'std'],
    'string_col': 'count'
})

# Pivot tables
pivot_table = df.pivot_table(values='value', index='category', columns='subcategory', aggfunc='mean')

# Cross-tabulation
cross_tab = pd.crosstab(df['category'], df['subcategory'])
          `}
        />
      </section>

      <section>
        <h2>Advanced Pandas Techniques</h2>
        <p>
          Here are some advanced techniques that can be useful in data
          manipulation:
        </p>
        <CodeBlock
          language="python"
          code={`
# Apply custom functions
df['new_col'] = df['col'].apply(lambda x: x * 2)

# Window functions
df['rolling_mean'] = df['value'].rolling(window=3).mean()

# Time series operations
df['month'] = df['date'].dt.month
df_resampled = df.resample('M', on='date').mean()

# String operations
df['lower_case'] = df['text'].str.lower()
df['contains_word'] = df['text'].str.contains('python')

# Categorical data
df['category'] = pd.Categorical(df['category'])
df['category_coded'] = df['category'].cat.codes

# Multi-index operations
df_multi = df.set_index(['date', 'category'])
df_unstacked = df_multi.unstack(level='category')
          `}
        />
      </section>

      <section>
        <h2>Best Practices for Data Manipulation</h2>
        <ul>
          <li>
            <strong>Vectorization:</strong> Use vectorized operations instead of
            loops when possible for better performance.
          </li>
          <li>
            <strong>Chaining:</strong> Use method chaining for cleaner and more
            readable code.
          </li>
          <li>
            <strong>Avoiding copies:</strong> Use inplace=True or assignment to
            avoid creating unnecessary copies of data.
          </li>
          <li>
            <strong>Memory management:</strong> Be aware of memory usage,
            especially when working with large datasets.
          </li>
          <li>
            <strong>Data types:</strong> Use appropriate data types (e.g.,
            categories for categorical data) to optimize memory usage and
            performance.
          </li>
          <li>
            <strong>Documentation:</strong> Document your data manipulation
            steps for reproducibility.
          </li>
        </ul>
      </section>
    </Container>
  );
};

export default DataManipulation;
