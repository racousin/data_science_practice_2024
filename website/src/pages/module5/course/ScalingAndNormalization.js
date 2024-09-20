import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from "components/DataInteractionPanel";
import { InlineMath, BlockMath } from "react-katex";

const ScalingAndNormalization = () => {
  const requirementsUrl =
    process.env.PUBLIC_URL + "/modules/module5/course/module5_requirements.txt";
  const DataUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/module5_course_scaling_and_normalization";
  const notebookUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/scaling_and_normalization.ipynb";
  const notebookHtmlUrl =
    process.env.PUBLIC_URL +
    "/modules/module5/course/scaling_and_normalization.html";
  const notebookColabUrl =
    process.env.PUBLIC_URL +
    "website/public/modules/module5/course/scaling_and_normalization.ipynb";
  const metadata = {
    description:
      "This dataset contains information about housing in California districts, derived from the 1990 census.",
    source: "California Housing Dataset from StatLib repository",
    target: "PRICE",
    listData: [
      {
        name: "MedInc",
        description:
          "Median income in block group (in tens of thousands of US Dollars)",
      },
      {
        name: "HouseAge",
        description: "Median house age in block group (in years)",
      },
      {
        name: "AveRooms",
        description: "Average number of rooms per household",
      },
      {
        name: "AveBedrms",
        description: "Average number of bedrooms per household",
      },
      {
        name: "Population",
        description: "Block group population",
      },
      {
        name: "AveOccup",
        description: "Average number of household members",
      },
      {
        name: "Latitude",
        description: "Block group latitude (in degrees)",
      },
      {
        name: "Longitude",
        description: "Block group longitude (in degrees)",
      },
      {
        name: "PRICE",
        description:
          "Median house value in block group (in hundreds of thousands of US Dollars)",
      },
    ],
  };
  return (
    <Container fluid>
      <h1 className="my-4">Scaling and Normalization</h1>

      <Row>
        <Col>
          <h2 id="why-scale-and-normalize">Why Scale and Normalize?</h2>
          <p>
            Different features of data often vary widely in magnitudes, units,
            and range. Without proper scaling or normalization:
          </p>
          <ul>
            <li>
              Algorithms that calculate distances between data points can be
              disproportionately influenced by one feature.
            </li>
            <li>
              Gradient descent-based algorithms may take longer to converge.
            </li>
            <li>
              Features with larger ranges could dominate the model's learning,
              leading to suboptimal performance.
            </li>
          </ul>

          <h2 id="scaling-methods">Scaling Methods</h2>
          <p>
            Scaling methods are used to transform features to a common scale
            without distorting differences in the ranges of values. Here are
            some common scaling methods:
          </p>

          <h3>1. Min-Max Scaling</h3>
          <p>
            Min-Max scaling (also known as normalization) scales features to a
            fixed range, typically 0 to 1.
          </p>
          <p>
            <strong>Formula:</strong>
          </p>
          <p>
            <InlineMath math="X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}" />
          </p>
          <p>
            Where X is the original value, and X_min and X_max are the minimum
            and maximum values of the feature.
          </p>
          <p>
            <strong>Characteristics:</strong>
          </p>
          <ul>
            <li>Preserves zero values in sparse data</li>
            <li>Preserves the shape of the original distribution</li>
            <li>Doesn't reduce the effects of outliers</li>
          </ul>

          <h3>2. Standard Scaling (Z-score Normalization)</h3>
          <p>
            Standard scaling transforms features to have zero mean and unit
            variance.
          </p>
          <p>
            <strong>Formula:</strong>
          </p>
          <p>
            <InlineMath math="X_{scaled} = \frac{X - \mu}{\sigma}" />
          </p>
          <p>
            Where X is the original value, μ is the mean of the feature, and σ
            is the standard deviation.
          </p>
          <p>
            <strong>Characteristics:</strong>
          </p>
          <ul>
            <li>
              Useful when features have different scales but approximately
              normal distributions
            </li>
            <li>Less affected by outliers compared to min-max scaling</li>
            <li>Does not bound values to a specific range</li>
          </ul>

          <h3>3. MaxAbs Scaling</h3>
          <p>
            MaxAbs scaling scales each feature by its maximum absolute value.
          </p>
          <p>
            <strong>Formula:</strong>
          </p>
          <p>
            <InlineMath math="X_{scaled} = \frac{X}{\max(|X|)}" />
          </p>
          <p>
            Where X is the original value and max(|X|) is the maximum absolute
            value of the feature.
          </p>
          <p>
            <strong>Characteristics:</strong>
          </p>
          <ul>
            <li>Scales the data to the range [-1, 1]</li>
            <li>Does not shift/center the data, thus preserving sparsity</li>
            <li>Less sensitive to outliers than min-max scaling</li>
          </ul>

          <h3>4. Robust Scaling</h3>
          <p>Robust scaling uses statistics that are robust to outliers.</p>
          <p>
            <strong>Formula:</strong>
          </p>
          <p>
            <InlineMath math="X_{scaled} = \frac{X - \text{median}(X)}{\text{IQR}(X)}" />
          </p>
          <p>
            Where X is the original value, median(X) is the median of the
            feature, and IQR(X) is the interquartile range.
          </p>
          <p>
            <strong>Characteristics:</strong>
          </p>
          <ul>
            <li>Less influenced by outliers</li>
            <li>Useful for data with outliers or skewed distributions</li>
            <li>Does not bound values to a specific range</li>
          </ul>

          <h2 id="normalization-methods">Normalization Methods</h2>
          <p>
            Normalization typically refers to adjusting values measured on
            different scales to a common scale, often scaling the length of
            vectors to unity.
          </p>

          <h3>1. L1 Normalization (Least Absolute Deviations)</h3>
          <p>
            L1 normalization ensures the sum of absolute values is 1 for each
            sample.
          </p>
          <p>
            <strong>Formula:</strong>
          </p>
          <p>
            <InlineMath math="X_{normalized} = \frac{X}{||X||_1}" />
          </p>
          <p>
            Where ||X||_1 is the L1 norm (sum of absolute values) of the sample.
          </p>
          <p>
            <strong>Characteristics:</strong>
          </p>
          <ul>
            <li>
              Commonly used in text classification and Natural Language
              Processing
            </li>
            <li>Produces sparse results where many features are zero</li>
          </ul>

          <h3>2. L2 Normalization (Least Squares)</h3>
          <p>
            L2 normalization ensures the sum of squares is 1 for each sample.
          </p>
          <p>
            <strong>Formula:</strong>
          </p>
          <p>
            <InlineMath math="X_{normalized} = \frac{X}{||X||_2}" />
          </p>
          <p>Where ||X||_2 is the L2 norm (Euclidean norm) of the sample.</p>
          <p>
            <strong>Characteristics:</strong>
          </p>
          <ul>
            <li>Commonly used when dealing with feature vectors</li>
            <li>
              Useful in algorithms sensitive to feature magnitude (e.g., neural
              networks)
            </li>
          </ul>

          <h3>3. Max Normalization</h3>
          <p>
            Max normalization scales the features by dividing each value by the
            maximum value for that feature.
          </p>
          <p>
            <strong>Formula:</strong>
          </p>
          <p>
            <InlineMath math="X_{normalized} = \frac{X}{\max(X)}" />
          </p>
          <p>Where max(X) is the maximum value of the feature.</p>
          <p>
            <strong>Characteristics:</strong>
          </p>
          <ul>
            <li>
              Scales features to the range [0, 1] or [-1, 1] if negative values
              are present
            </li>
            <li>Preserves zero entries in sparse data</li>
          </ul>

          <h2 id="choosing-method">Choosing the Right Method</h2>
          <p>
            The choice between scaling and normalization methods depends on the
            specific requirements of your data and the algorithm you're using:
          </p>
          <ul>
            <li>
              Use Min-Max scaling when you need bounded values and your data
              doesn't have significant outliers.
            </li>
            <li>
              Use Standard scaling when you assume your data follows a normal
              distribution and when dealing with algorithms sensitive to feature
              magnitude (e.g., neural networks, SVMs).
            </li>
            <li>
              Use MaxAbs scaling when dealing with sparse data and you want to
              preserve zero values.
            </li>
            <li>Use Robust scaling when your data contains many outliers.</li>
            <li>
              Use L1 normalization for sparse data or when you want many
              features to be zero.
            </li>
            <li>
              Use L2 normalization when the magnitude of vectors is important
              (e.g., in cosine similarity).
            </li>
          </ul>

          <p>
            It's often beneficial to experiment with different scaling and
            normalization techniques to determine which works best for your
            specific dataset and machine learning task.
          </p>

          <CodeBlock
            language={"python"}
            code={`from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer

# Min-Max Scaling
scaler = MinMaxScaler()
data_minmax = scaler.fit_transform(data)

# Standard Scaling
scaler = StandardScaler()
data_standard = scaler.fit_transform(data)

# MaxAbs Scaling
scaler = MaxAbsScaler()
data_maxabs = scaler.fit_transform(data)

# Robust Scaling
scaler = RobustScaler()
data_robust = scaler.fit_transform(data)

# L1 Normalization
normalizer = Normalizer(norm='l1')
data_l1 = normalizer.fit_transform(data)

# L2 Normalization
normalizer = Normalizer(norm='l2')
data_l2 = normalizer.fit_transform(data)`}
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

export default ScalingAndNormalization;
