import { NavLink, Outlet } from "react-router-dom";
import "./Layout.css";

const NAV_ITEMS = [
  { to: "/model", label: "模型訓練" },
  { to: "/quant", label: "量化回測" },
];

export default function Layout() {
  return (
    <div className="app-scroll">
      <header className="shell-header">
        <svg
          className="shell-lockup"
          viewBox="0 0 700 152"
          fill="none"
          role="img"
          aria-label="Hermesnote"
        >
          <g transform="translate(0 6)">
            <rect x="0" y="14" width="24" height="126" rx="5" fill="#F6F4F0" />
            <line x1="38" y1="72" x2="38" y2="78" stroke="#F6F4F0" strokeWidth="2.2" />
            <line x1="38" y1="106" x2="38" y2="120" stroke="#F6F4F0" strokeWidth="2.2" />
            <rect x="30" y="78" width="16" height="28" fill="#F6F4F0" />
            <line x1="62" y1="50" x2="62" y2="64" stroke="#F6F4F0" strokeWidth="2.2" />
            <line x1="62" y1="72" x2="62" y2="92" stroke="#F6F4F0" strokeWidth="2.2" />
            <rect x="54" y="64" width="16" height="8" fill="#F6F4F0" />
            <rect x="78" y="48" width="16" height="7" fill="#E2624F" />
            <line x1="86" y1="55" x2="86" y2="100" stroke="#E2624F" strokeWidth="2.2" />
            <rect x="100" y="0" width="24" height="126" rx="5" fill="#E2624F" />
          </g>
          <text
            x="150"
            y="106"
            fontFamily="Archivo, Helvetica, Arial, sans-serif"
            fontSize="106"
            fontWeight="700"
            letterSpacing="-3.7"
            fill="#F6F4F0"
          >
            Hermes
            <tspan fontWeight="500" fill="rgba(246,244,240,0.72)">
              note
            </tspan>
          </text>
        </svg>

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
