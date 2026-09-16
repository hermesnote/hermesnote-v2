import { Link, NavLink, Outlet } from "react-router-dom";
import LogoMark from "./LogoMark";
import "./Layout.css";

const NAV_ITEMS = [
  { to: "/model", label: "模型訓練" },
  { to: "/quant", label: "量化回測" },
  { to: "/hermes", label: "Hermes" },
];

export default function Layout() {
  return (
    <div className="app-scroll">
      <header className="shell-header">
        <Link to="/" aria-label="回首頁">
          <LogoMark className="shell-lockup" />
        </Link>

        <nav className="shell-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                "shell-nav-link" + (isActive ? " is-active" : "")
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      <main className="shell-main">
        <Outlet />
      </main>
    </div>
  );
}
