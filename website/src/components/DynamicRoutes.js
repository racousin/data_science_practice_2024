import React, { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import ScrollToTop from "components/ScrollToTop";

const DynamicRoutes = ({ routes }) => (
  <Suspense fallback={<div>Loading...</div>}>
    <ScrollToTop />
    <Routes>
      {routes.map(({ to, component: Component }) => (
        <Route key={to} path={to} element={<Component />} />
      ))}
    </Routes>
  </Suspense>
);

export default DynamicRoutes;
