import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const ContentBasedFiltering = () => {
  return (
    <Container>
      <h1>Content-Based Filtering</h1>

      <section id="definition">
        <h2>Definition</h2>
        <p>
          Content-based filtering is a recommendation technique that uses item
          features to suggest similar items to users based on their previous
          preferences. This method creates a profile for each item and user,
          then uses similarity measures to match users with items.
        </p>
      </section>

      <section id="item-representation">
        <h2>Item Representation</h2>
        <p>
          Items are represented as feature vectors. For example, a movie could
          be represented as:
        </p>
        <CodeBlock
          language="python"
          code={`
movie = {
  'id': 1,
  'title': 'The Matrix',
  'genre': ['sci-fi', 'action'],
  'director': 'Wachowski',
  'year': 1999,
  'keywords': ['virtual reality', 'dystopia', 'AI']
}
          `}
        />
      </section>

      <section id="user-profiles">
        <h2>User Profiles</h2>
        <p>
          User profiles are typically constructed by aggregating the features of
          items the user has interacted with. This can be represented as a
          weighted average of item features:
        </p>
        <BlockMath>
          {`
user\\_profile = \\frac{\\sum_{i} w_i \\cdot item\\_i\\_features}{\\sum_{i} w_i}
          `}
        </BlockMath>
        <p>
          where <InlineMath>w_i</InlineMath> is the weight (e.g., rating) given
          by the user to item i
        </p>
      </section>

      <section id="similarity-measures">
        <h2>Similarity Measures</h2>
        <p>
          Cosine similarity is commonly used to measure the similarity between
          user profiles and item features:
        </p>
        <BlockMath>
          {`
cosine\\_similarity(A, B) = \\frac{A \\cdot B}{\\|A\\| \\cdot \\|B\\|}
          `}
        </BlockMath>
        <p>
          where A and B are feature vectors, · denotes dot product, and ||x|| is
          the magnitude of vector x
        </p>
      </section>

      <section id="implementing-content-based">
        <h2>Implementing Content-Based Filtering</h2>
        <p>Here's a basic implementation using Python and scikit-learn:</p>
        <CodeBlock
          language="python"
          code={`
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Assume we have a list of item descriptions
item_descriptions = ["sci-fi movie about virtual reality", 
                     "action movie with car chases",
                     "romantic comedy set in Paris"]

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(item_descriptions)

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(item_index, cosine_sim=cosine_sim):
    sim_scores = list(enumerate(cosine_sim[item_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:] # Exclude the item itself

# Get recommendations for the first item
recommendations = get_recommendations(0)
print(recommendations)
          `}
        />
      </section>
    </Container>
  );
};

export default ContentBasedFiltering;
