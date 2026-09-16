import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import LogoMark from "../../components/LogoMark";
import { useAuth } from "../../auth/AuthContext";
import "./AdminLayout.css";

const NAV_ITEMS = [
  { to: "/admin/trading", label: "交易系統" },
  { to: "/admin/model", label: "模型訓練" },
  { to: "/admin/quant", label: "量化回測" },
  { to: "/admin/ledger", label: "家庭理財" },
];

export default function AdminLayout() {
  const { email, picture, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  return (
    <div className="admin-shell">
      <div className="admin-shell-bar">
        <LogoMark className="admin-shell-logo" />

        <div className="admin-shell-menu" ref={menuRef}>
          <button className="admin-shell-avatar-btn" onClick={() => setMenuOpen((v) => !v)} aria-label="帳號選單">
            {picture ? (
              <img className="admin-shell-avatar" src={picture} alt={email ?? ""} referrerPolicy="no-referrer" />
            ) : (
              <div className="admin-shell-avatar admin-shell-avatar-fallback">{email?.[0]?.toUpperCase()}</div>
            )}
          </button>
          {menuOpen && (
            <div className="admin-shell-dropdown">
              <div className="admin-shell-dropdown-email">{email}</div>
              <button className="admin-shell-dropdown-item" disabled>
                帳號設定
              </button>
              <button className="admin-shell-dropdown-item" onClick={logout}>
                登出
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="admin-shell-body">
        <nav className="admin-shell-sidebar">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => "admin-shell-sidebar-link" + (isActive ? " is-active" : "")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <main className="admin-shell-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
