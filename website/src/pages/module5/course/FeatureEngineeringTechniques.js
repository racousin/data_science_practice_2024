import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";

const FeatureEngineeringTechniques = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_feature_engineering.csv";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/feature_engineering.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/feature_engineering.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/feature_engineering.ipynb";
  const metadata = {
    description:
      "This dataset contains hourly weather and holiday information along with corresponding bike rental counts, aimed at predicting bike rental demand based on various environmental and temporal factors.",
    source: "City Bike Share Program and Local Weather Station",
    target: "count",
    listData: [
      {
        name: "datetime",
        description:
          "Date and time of the observation, typically in hourly intervals.",
        dataType: "Datetime",
        example: "2022-01-01 13:00:00",
      },
      {
        name: "holiday",
        description:
          "Boolean indicating whether the day is a holiday (1) or not (0).",
        dataType: "Boolean",
        example: "1 (holiday), 0 (not holiday)",
      },
      {
        name: "temp",
        description: "Temperature in Celsius at the time of observation.",
        dataType: "Continuous",
        example: "25.5",
      },
      {
        name: "humidity",
        description: "Relative humidity as a percentage.",
        dataType: "Continuous",
        example: "65.0",
      },
      {
        name: "windspeed",
        description: "Wind speed in km/h at the time of observation.",
        dataType: "Continuous",
        example: "12.7",
      },
      {
        name: "pressure",
        description: "Atmospheric pressure in hectopascals (hPa).",
        dataType: "Continuous",
        example: "1015.2",
      },
      {
        name: "count",
        description: "Number of bikes rented in the given hour.",
        dataType: "Discrete",
        example: "145",
      },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">Feature Engineering Techniques</h1>
      <p>
        Feature engineering transforms raw data into formats that are better
        suited for models, enhancing their performance by incorporating
        domain-specific knowledge and data insights. This process is crucial for
        tailoring data to the specific needs of the business and predictive
        models.
      </p>

      <Row>
        <Col>
          <h2 id="decomposition">Decomposition</h2>
          <p>
            Breaking down complex features into simpler components can reveal
            additional insights. For example, extracting date parts from a
            timestamp can help isolate seasonal trends and cyclic behavior in
            data.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd
# Example: Extracting components from a datetime column
df['date'] = pd.to_datetime(df['timestamp'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['second'] = df['date'].dt.second
df['weekday'] = df['date'].dt.weekday`}
          />
          <h2 id="binning">Binning</h2>
          <p>
            Binning is used to transform numerical data into categorical labels,
            which can sometimes improve model stability and performance,
            particularly with non-linear models.
          </p>
          <CodeBlock
            language={"python"}
            code={`import pandas as pd
# Example of binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['child', 'young adult', 'adult', 'senior'])`}
          />
          <h2 id="interaction-features">Creation of Interaction Features</h2>
          <p>
            Interaction features involve combinations of two or more features.
            This can uncover relationships between variables that are not
            apparent when considered alone.
          </p>
          <CodeBlock
            language={"python"}
            code={`# Example of creating interaction features
df['price_per_item'] = df['total_price'] / df['quantity']`}
          />

          <h2 id="aggregation">Aggregation</h2>
          <p>
            Aggregating data provides summary statistics for groups, which is
            especially useful in understanding grouped data behavior, such as
            customer segmentation.
          </p>
          <CodeBlock
            language={"python"}
            code={`# Example of aggregation
grouped_data = df.groupby('category').agg({'price': ['mean', 'sum', 'max']})`}
          />
          <h2 id="polynomial-features">Polynomial Features</h2>
          <p>
            Generating polynomial and interaction features can expose more
            complex relationships between features, which linear models can then
            leverage.
          </p>
          <CodeBlock
            language={"python"}
            code={`from sklearn.preprocessing import PolynomialFeatures
# Generating polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
df_poly = poly.fit_transform(df[['feature1', 'feature2']])
df_poly = pd.DataFrame(df_poly, columns=poly.get_feature_names(['feature1', 'feature2']))`}
          />
          <h2 id="transformation">Feature Transformation</h2>
          <p>
            Applying mathematical transformations to data to scale, normalize,
            or change its distribution can significantly influence model
            accuracy.
          </p>
          <CodeBlock
            language={"python"}
            code={`from sklearn.preprocessing import PowerTransformer
transformer = PowerTransformer(method='yeo-johnson')
df['normalized_data'] = transformer.fit_transform(df[['data']])`}
          />

          <h2 id="time-series">Time-series Transformations</h2>
          <p>
            Time-series data can be transformed using techniques such as window
            functions and lag features to model trends and seasonality
            effectively.
          </p>
          <CodeBlock
            language={"python"}
            code={`# Example of lag features
df['lag1'] = df['feature'].shift(1)
df['lag2'] = df['feature'].shift(2)`}
          />
        </Col>
      </Row>
      <Row>
        <div id="notebook-example"></div>
        <DataInteractionPanel
          DataUrl={DataUrl}
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          requirementsUrl={requirementsUrl}
          metadata={metadata}
        />
      </Row>
    </Container>
  );
};

export default FeatureEngineeringTechniques;
