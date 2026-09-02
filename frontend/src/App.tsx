import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import QuantBacktestPage from "./pages/QuantBacktestPage";
import ModelTrainingPage from "./pages/ModelTrainingPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Navigate to="/quant" replace />} />
          <Route path="/quant" element={<QuantBacktestPage />} />
          <Route path="/model" element={<ModelTrainingPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
