import React, { useEffect, useRef } from "react";
import * as d3 from "d3";

const DataSplitDiagram = () => {
  const svgRef = useRef();

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const width = 600;
    const height = 300;

    svg.attr("width", width).attr("height", height);

    // Define arrow marker
    svg
      .append("defs")
      .append("marker")
      .attr("id", "arrowhead")
      .attr("viewBox", "-0 -5 10 10")
      .attr("refX", 8)
      .attr("refY", 0)
      .attr("orient", "auto")
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("xoverflow", "visible")
      .append("svg:path")
      .attr("d", "M 0,-5 L 10 ,0 L 0,5")
      .attr("fill", "#000")
      .style("stroke", "none");

    // Draw rectangles
    const rectangles = [
      { x: 50, y: 100, width: 120, height: 60, text: "Dataset" },
      { x: 250, y: 110, width: 80, height: 40, text: "Split" },
      { x: 410, y: 50, width: 140, height: 60, text: "Training Set" },
      { x: 410, y: 150, width: 140, height: 60, text: "Test Set" },
    ];

    svg
      .selectAll("rect")
      .data(rectangles)
      .enter()
      .append("rect")
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("fill", "none")
      .attr("stroke", "#000")
      .attr("stroke-width", 2)
      .attr("rx", 10)
      .attr("ry", 10);

    // Add text to rectangles
    svg
      .selectAll("text")
      .data(rectangles)
      .enter()
      .append("text")
      .attr("x", (d) => d.x + d.width / 2)
      .attr("y", (d) => d.y + d.height / 2)
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .text((d) => d.text);

    // Draw arrows
    const arrows = [
      { x1: 170, y1: 130, x2: 250, y2: 130 },
      { x1: 330, y1: 130, x2: 410, y2: 80 },
      { x1: 330, y1: 130, x2: 410, y2: 180 },
    ];

    svg
      .selectAll("line")
      .data(arrows)
      .enter()
      .append("line")
      .attr("x1", (d) => d.x1)
      .attr("y1", (d) => d.y1)
      .attr("x2", (d) => d.x2)
      .attr("y2", (d) => d.y2)
      .attr("stroke", "#000")
      .attr("stroke-width", 2)
      .attr("marker-end", "url(#arrowhead)");
  }, []);

  return <svg ref={svgRef}></svg>;
};

export default DataSplitDiagram;
