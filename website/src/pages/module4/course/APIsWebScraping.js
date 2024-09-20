import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const APIsWebScraping = () => {
  return (
    <Container fluid>
      <h1 className="my-4">APIs and Web Scraping</h1>

      <section>
        <h2 id="restful-apis">RESTful APIs</h2>
        <p>
          RESTful APIs (Representational State Transfer) are a standard way for
          systems to communicate over the internet. They allow you to retrieve
          and manipulate data from web services.
        </p>

        <h3>Authentication Methods</h3>
        <p>
          Most APIs require authentication to ensure secure access to data.
          Common authentication methods include:
        </p>
        <ul>
          <li>API Keys</li>
          <li>OAuth</li>
          <li>JSON Web Tokens (JWT)</li>
          <li>Basic Authentication</li>
        </ul>

        <h3>Making API Requests with Python (requests library)</h3>
        <p>
          We'll use the JSONPlaceholder API
          (https://jsonplaceholder.typicode.com) for our examples. This is a
          free fake API for testing and prototyping.
        </p>
        <p>
          The API uses the HTTP protocol. HTTP (Hypertext Transfer Protocol) is
          the foundation of data communication on the web. HTTPS is the secure
          version of HTTP, encrypting the data sent between the client and the
          server.
        </p>
        <CodeBlock
          language="python"
          code={`
import requests

# GET request to fetch a post
response = requests.get('https://jsonplaceholder.typicode.com/posts/1')
print("GET Response:")
print(response.json())

# POST request to create a new post
new_post = {
    'title': 'New Post',
    'body': 'This is a new post.',
    'userId': 1
}
post_response = requests.post('https://jsonplaceholder.typicode.com/posts', json=new_post)
print("\nPOST Response:")
print(post_response.json())

# Handling errors
try:
    response = requests.get('https://jsonplaceholder.typicode.com/nonexistent')
    response.raise_for_status()
    data = response.json()
except requests.exceptions.HTTPError as err:
    print(f"\nHTTP error occurred: {err}")
except requests.exceptions.RequestException as err:
    print(f"\nAn error occurred: {err}")
  `}
        />

        <h3>Parsing JSON Responses</h3>
        <p>
          Most APIs return data in JSON format. Python's requests library
          automatically parses JSON responses, but understanding how to work
          with JSON data is crucial.
        </p>
        <CodeBlock
          language="python"
          code={`
# Fetch multiple posts
response = requests.get('https://jsonplaceholder.typicode.com/posts')
data = response.json()

# Access specific data
first_post_title = data[0]['title']
print(f"Title of first post: {first_post_title}")

# Handle potential KeyError
first_post_author = data[0].get('author', 'Unknown')
print(f"Author of first post: {first_post_author}")

# Convert JSON to pandas DataFrame
import pandas as pd
df = pd.DataFrame(data)
print("\nFirst 5 posts:")
print(df[['id', 'title']].head())
  `}
        />
      </section>

      <section>
        <h2 id="web-scraping">Web Scraping Basics</h2>
        <p>
          Web scraping is the process of extracting data from websites. It's
          useful when data is not available through an API. We'll use the page
          at https://www.raphaelcousin.com/page-to-scrap as our example.
        </p>

        <h3>HTML Structure and CSS Selectors</h3>
        <p>
          Understanding HTML structure and CSS selectors is crucial for
          effective web scraping. Let's look at a part of our example page:
        </p>
        <CodeBlock
          language="html"
          code={`
<div class="scraping-example">
  <h1 id="main-title">Web Scraping Practice Page</h1>
  <div class="article">
    <h2>Article 1</h2>
    <p class="author">By John Doe</p>
    <p class="content">This is the content of the first article.</p>
    <span class="date">2024-07-11</span>
  </div>
  <!-- More content... -->
</div>
    `}
        />
        <p>CSS Selectors for this structure:</p>
        <ul>
          <li>
            <code>.scraping-example</code>: selects the main container
          </li>
          <li>
            <code>#main-title</code>: selects the main title
          </li>
          <li>
            <code>.article</code>: selects all article divs
          </li>
          <li>
            <code>.article h2</code>: selects article titles
          </li>
          <li>
            <code>.author</code>: selects author paragraphs
          </li>
        </ul>

        <h3>Using BeautifulSoup for Scraping</h3>
        <p>
          BeautifulSoup is a popular Python library for web scraping. Here's how
          to use it:
        </p>
        <CodeBlock
          language="python"
          code={`
import requests
from bs4 import BeautifulSoup

url = 'https://www.raphaelcousin.com/page-to-scrap'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract main title
title = soup.find(id='main-title').text

# Extract articles
articles = []
for article in soup.find_all('div', class_='article'):
    articles.append({
        'title': article.find('h2').text,
        'author': article.find('p', class_='author').text,
        'content': article.find('p', class_='content').text,
        'date': article.find('span', class_='date').text
    })

print(f"Main Title: {title}")
for article in articles:
    print(f"\nArticle: {article['title']}")
    print(f"Author: {article['author']}")
    print(f"Content: {article['content']}")
    print(f"Date: {article['date']}")
    `}
        />

        <h3>Ethical Considerations and Legality</h3>
        <p>
          When web scraping, it's important to consider ethical and legal
          implications:
        </p>
        <ul>
          <li>Respect robots.txt files and website terms of service</li>
          <li>Don't overload servers with too many requests</li>
          <li>Be aware of copyright and data protection laws</li>
          <li>Consider using APIs if available instead of scraping</li>
          <li>Identify your scraper in the user-agent string</li>
        </ul>
        <CodeBlock
          language="python"
          code={`
import requests
from urllib.robotparser import RobotFileParser

url = "https://www.raphaelcousin.com"
rp = RobotFileParser()
rp.set_url(f"{url}/robots.txt")
rp.read()

if rp.can_fetch("*", f"{url}/page-to-scrap"):
    headers = {
        'User-Agent': 'YourCompany DataScienceBot 1.0',
    }
    response = requests.get(f"{url}/page-to-scrap", headers=headers)
    # Process the response...
else:
    print("Scraping is not allowed for this page")
    `}
        />
      </section>
    </Container>
  );
};

export default APIsWebScraping;
