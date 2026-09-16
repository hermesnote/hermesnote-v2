import type { ReactNode } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "./AuthContext";

export default function RequireAuth({ children }: { children: ReactNode }) {
  const { status } = useAuth();

  if (status === "checking") {
    return <div className="admin-status">確認登入狀態中…</div>;
  }
  if (status === "anon") {
    return <Navigate to="/admin/login" replace />;
  }
  return <>{children}</>;
}
