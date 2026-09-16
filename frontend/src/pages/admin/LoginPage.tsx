import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import "./LoginPage.css";

const API_BASE = import.meta.env.VITE_API_BASE as string;
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID as string;

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: {
            client_id: string;
            callback: (response: { credential: string }) => void;
          }) => void;
          renderButton: (parent: HTMLElement, options: Record<string, string>) => void;
        };
      };
    };
  }
}

export default function LoginPage() {
  const buttonRef = useRef<HTMLDivElement>(null);
  const { login } = useAuth();
  const navigate = useNavigate();
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function handleCredentialResponse(response: { credential: string }) {
      try {
        const res = await fetch(`${API_BASE}/api/auth/google`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ credential: response.credential }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body.detail || `登入失敗（${res.status}）`);
        }
        const json: { access_token: string; email: string; picture: string } = await res.json();
        login(json.access_token, json.email, json.picture);
        navigate("/admin", { replace: true });
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : "登入失敗");
      }
    }

    function tryInit() {
      if (cancelled) return;
      if (!window.google || !buttonRef.current) {
        setTimeout(tryInit, 100);
        return;
      }
      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: handleCredentialResponse,
      });
      window.google.accounts.id.renderButton(buttonRef.current, {
        theme: "outline",
        size: "large",
      });
    }

    // 只有這個頁面需要 Google 登入，腳本也只在這裡動態載入，
    // 不放在 index.html 讓每個公開頁面都載入用不到的第三方腳本。
    const existing = document.getElementById("google-gsi-script");
    if (existing) {
      tryInit();
    } else {
      const script = document.createElement("script");
      script.id = "google-gsi-script";
      script.src = "https://accounts.google.com/gsi/client";
      script.async = true;
      script.defer = true;
      script.onload = tryInit;
      document.head.appendChild(script);
    }

    return () => {
      cancelled = true;
    };
  }, [login, navigate]);

  return (
    <div className="admin-login">
      <div className="admin-login-box">
        <div className="admin-login-title">hermesnote 後台</div>
        <div ref={buttonRef} className="admin-login-btn" />
        {errorMsg && <div className="admin-login-error">{errorMsg}</div>}
      </div>
    </div>
  );
}
