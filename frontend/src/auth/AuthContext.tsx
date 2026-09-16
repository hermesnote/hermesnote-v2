import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

const API_BASE = import.meta.env.VITE_API_BASE as string;
const TOKEN_KEY = "hermesnote_admin_token";

type AuthState = {
  email: string | null;
  picture: string | null;
  status: "checking" | "authed" | "anon";
  login: (token: string, email: string, picture: string) => void;
  logout: () => void;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [email, setEmail] = useState<string | null>(null);
  const [picture, setPicture] = useState<string | null>(null);
  const [status, setStatus] = useState<"checking" | "authed" | "anon">("checking");

  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      setStatus("anon");
      return;
    }
    fetch(`${API_BASE}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("未登入");
        return res.json();
      })
      .then((json: { email: string; picture: string }) => {
        setEmail(json.email);
        setPicture(json.picture);
        setStatus("authed");
      })
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        setStatus("anon");
      });
  }, []);

  function login(token: string, email: string, picture: string) {
    localStorage.setItem(TOKEN_KEY, token);
    setEmail(email);
    setPicture(picture);
    setStatus("authed");
  }

  function logout() {
    localStorage.removeItem(TOKEN_KEY);
    setEmail(null);
    setPicture(null);
    setStatus("anon");
  }

  return (
    <AuthContext.Provider value={{ email, picture, status, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth 必須在 AuthProvider 內使用");
  return ctx;
}

export function getAuthToken() {
  return localStorage.getItem(TOKEN_KEY);
}
