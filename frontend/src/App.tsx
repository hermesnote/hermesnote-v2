import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import QuantBacktestPage from "./pages/QuantBacktestPage";
import ModelTrainingPage from "./pages/ModelTrainingPage";
import HermesResumePage from "./pages/HermesResumePage";
import LoginPage from "./pages/admin/LoginPage";
import AdminLayout from "./pages/admin/AdminLayout";
import AdminDashboard from "./pages/admin/AdminDashboard";
import ModelSettings from "./pages/admin/sections/ModelSettings";
import QuantSettings from "./pages/admin/sections/QuantSettings";
import TradingSystem from "./pages/admin/sections/TradingSystem";
import Ledger from "./pages/admin/sections/Ledger";
import { AuthProvider } from "./auth/AuthContext";
import RequireAuth from "./auth/RequireAuth";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="/quant" element={<QuantBacktestPage />} />
            <Route path="/model" element={<ModelTrainingPage />} />
            <Route path="/hermes" element={<HermesResumePage />} />
          </Route>

          <Route path="/admin/login" element={<LoginPage />} />
          <Route
            path="/admin"
            element={
              <RequireAuth>
                <AdminLayout />
              </RequireAuth>
            }
          >
            <Route index element={<AdminDashboard />} />
            <Route path="model" element={<ModelSettings />} />
            <Route path="quant" element={<QuantSettings />} />
            <Route path="trading" element={<TradingSystem />} />
            <Route path="ledger" element={<Ledger />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
