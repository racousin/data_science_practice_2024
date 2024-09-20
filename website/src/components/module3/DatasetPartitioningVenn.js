import React from "react";
import { VennDiagram } from "venn.js";
import { useEffect, useRef } from "react";
import * as d3 from "d3";

const DatasetPartitioningVenn = () => {
  const vennRef = useRef(null);

  useEffect(() => {
    const sets = [
      { sets: ["D_train"], size: 70 },
      { sets: ["D_val"], size: 15 },
      { sets: ["D_test"], size: 15 },
    ];

    const chart = VennDiagram().width(300).height(300);

    d3.select(vennRef.current).datum(sets).call(chart);

    return () => {
      d3.select(vennRef.current).selectAll("*").remove();
    };
  }, []);

  return <div ref={vennRef}></div>;
};

export default DatasetPartitioningVenn;
