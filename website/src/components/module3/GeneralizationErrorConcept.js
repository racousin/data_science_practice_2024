import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const GeneralizationErrorConcept = () => {
  const data = [
    { complexity: 1, training: 0.5, test: 0.6 },
    { complexity: 2, training: 0.4, test: 0.5 },
    { complexity: 3, training: 0.3, test: 0.4 },
    { complexity: 4, training: 0.2, test: 0.3 },
    { complexity: 5, training: 0.1, test: 0.25 },
    { complexity: 6, training: 0.05, test: 0.3 },
    { complexity: 7, training: 0.02, test: 0.4 },
    { complexity: 8, training: 0.01, test: 0.5 },
  ];

  return (
    <div style={{ width: "100%", height: 300 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="complexity"
            label={{ value: "Model Complexity", position: "bottom" }}
          />
          <YAxis label={{ value: "Error", angle: -90, position: "left" }} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="training"
            stroke="#8884d8"
            name="Training Error"
          />
          <Line
            type="monotone"
            dataKey="test"
            stroke="#82ca9d"
            name="Test Error"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default GeneralizationErrorConcept;
