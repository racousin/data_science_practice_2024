import React from "react";

const PageToScrap = () => {
  return (
    <div className="scraping-example">
      <h1 id="main-title">Web Scraping Practice Page</h1>

      <div className="article">
        <h2>Article 1</h2>
        <p className="author">By John Doe</p>
        <p className="content">
          This is the content of the first article. It contains some text that
          can be scraped.
        </p>
        <span className="date">2024-07-11</span>
      </div>

      <div className="article">
        <h2>Article 2</h2>
        <p className="author">By Jane Smith</p>
        <p className="content">
          Here's another article with different content. Web scraping can
          extract this text too.
        </p>
        <span className="date">2024-07-10</span>
      </div>

      <ul className="item-list">
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
      </ul>

      <table className="data-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Age</th>
            <th>City</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Alice</td>
            <td>30</td>
            <td>New York</td>
          </tr>
          <tr>
            <td>Bob</td>
            <td>25</td>
            <td>London</td>
          </tr>
        </tbody>
      </table>

      <a href="https://example.com" className="external-link">
        External Link
      </a>
    </div>
  );
};

export default PageToScrap;
