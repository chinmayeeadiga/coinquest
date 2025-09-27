import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/home.jsx";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import AppLayout from "./layouts/app-layout";
import Crypto from "./pages/crypto";
import Trade from "./pages/trade";


const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      {
        path: "/",
        element: <Home />,
      },
      {
        path: "/crypto",
        element: <Crypto />,
      },
      {
        path: "/trade",
        element: <Trade />,
      },
    ],
  },
]);

function App() {
 return <RouterProvider router={router} />;
}

export default App;