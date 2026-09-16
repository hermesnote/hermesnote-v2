import { useState, type ReactElement } from "react";
import "./Hero.css";

type ViewKey = "trade" | "train" | "backtest";

const TABS: { key: ViewKey; title: string; desc: string }[] = [
  { key: "trade", title: "交易系統", desc: "模型訊號如何走到下單邏輯。" },
  { key: "train", title: "模型訓練", desc: "訓練過程即時渲染，loss 怎麼收斂。" },
  { key: "backtest", title: "量化回測", desc: "策略怎麼被驗證，walk-forward 逐 fold 推進。" },
];

function Clock() {
  const [now, setNow] = useState(new Date());
  useState(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  });
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    <span className="panel-clock">
      {pad(now.getHours())}:{pad(now.getMinutes())}:{pad(now.getSeconds())}
    </span>
  );
}

function TradeChart() {
  return (
    <svg viewBox="0 0 400 220" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <pattern id="grid-trade" width="40" height="25" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 25" fill="none" stroke="var(--panel-grid)" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="400" height="220" fill="url(#grid-trade)" />
      <g strokeWidth="1.6" opacity="0.85">
        <line x1="30" y1="130" x2="30" y2="165" stroke="var(--candle-up)" />
        <rect x="26" y="118" width="8" height="12" fill="var(--candle-up)" />
        <line x1="55" y1="110" x2="55" y2="150" stroke="var(--candle-down)" />
        <rect x="51" y="122" width="8" height="16" fill="var(--candle-down)" />
        <line x1="80" y1="100" x2="80" y2="135" stroke="var(--candle-up)" />
        <rect x="76" y="105" width="8" height="18" fill="var(--candle-up)" />
        <line x1="105" y1="95" x2="105" y2="128" stroke="var(--candle-up)" />
        <rect x="101" y="98" width="8" height="14" fill="var(--candle-up)" />
        <line x1="130" y1="105" x2="130" y2="140" stroke="var(--candle-down)" />
        <rect x="126" y="110" width="8" height="15" fill="var(--candle-down)" />
        <line x1="155" y1="90" x2="155" y2="125" stroke="var(--candle-up)" />
        <rect x="151" y="95" width="8" height="16" fill="var(--candle-up)" />
      </g>
      <line x1="180" y1="116" x2="180" y2="200" stroke="var(--line)" strokeWidth="1" strokeDasharray="3 4" />
    </svg>
  );
}

function TrainChart() {
  return (
    <svg viewBox="0 0 400 220" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <pattern id="grid-train" width="40" height="25" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 25" fill="none" stroke="var(--panel-grid)" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="400" height="220" fill="url(#grid-train)" />
      <polyline
        points="10,190 40,175 70,182 100,150 130,158 160,120 190,132 220,95 250,105 280,78 310,88 340,60"
        fill="none"
        stroke="var(--ink-faint)"
        strokeWidth="1.6"
        opacity="0.6"
      />
      <polyline
        points="10,205 40,198 70,192 100,178 130,170 160,150 190,140 220,118 250,108 280,90 310,80 340,62"
        fill="none"
        stroke="var(--candle-up)"
        strokeWidth="2"
      />
      <circle cx="340" cy="62" r="3" fill="var(--candle-up)" />
    </svg>
  );
}

function BacktestChart() {
  return (
    <svg viewBox="0 0 400 220" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <pattern id="grid-backtest" width="40" height="25" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 25" fill="none" stroke="var(--panel-grid)" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="400" height="220" fill="url(#grid-backtest)" />
      <polyline
        points="10,180 40,178 70,165 100,168 130,145 160,150 190,120 220,125 250,95 280,100 310,70 340,75 370,50"
        fill="none"
        stroke="var(--candle-up)"
        strokeWidth="2"
      />
      <polygon
        points="10,180 40,178 70,165 100,168 130,145 160,150 190,120 220,125 250,95 280,100 310,70 340,75 370,50 370,210 10,210"
        fill="var(--candle-up)"
        opacity="0.08"
      />
    </svg>
  );
}

const CHARTS: Record<ViewKey, () => ReactElement> = {
  trade: TradeChart,
  train: TrainChart,
  backtest: BacktestChart,
};

export default function Hero() {
  const [active, setActive] = useState<ViewKey>("trade");
  const ActiveChart = CHARTS[active];

  return (
    <section className="hero-viewport">
      <header className="identity">
        <div className="identity-lockup-wrap">
          <svg
            className="identity-lockup"
            viewBox="0 0 760 152"
            fill="none"
            role="img"
            aria-label="Hermesnote"
          >
            <g transform="translate(0 6)">
              <rect x="0" y="14" width="24" height="126" rx="5" fill="#17161A" />
              <line x1="38" y1="72" x2="38" y2="78" stroke="#17161A" strokeWidth="2.2" />
              <line x1="38" y1="106" x2="38" y2="120" stroke="#17161A" strokeWidth="2.2" />
              <rect x="30" y="78" width="16" height="28" fill="#17161A" />
              <line x1="62" y1="50" x2="62" y2="64" stroke="#17161A" strokeWidth="2.2" />
              <line x1="62" y1="72" x2="62" y2="92" stroke="#17161A" strokeWidth="2.2" />
              <rect x="54" y="64" width="16" height="8" fill="#17161A" />
              <rect x="78" y="48" width="16" height="7" fill="#BF3A2B" />
              <line x1="86" y1="55" x2="86" y2="100" stroke="#BF3A2B" strokeWidth="2.2" />
              <rect x="100" y="0" width="24" height="126" rx="5" fill="#BF3A2B" />
            </g>
            <text
              x="150"
              y="106"
              fontFamily="Archivo, Helvetica, Arial, sans-serif"
              fontSize="106"
              fontWeight="700"
              letterSpacing="-3.7"
              fill="#17161A"
            >
              Hermes
              <tspan fontWeight="500" fill="rgba(23,22,26,0.7)">
                note
              </tspan>
            </text>
          </svg>
        </div>
      </header>

      <div className="hero">
        <div className="hero-content">
        <div className="view-tabs-wrap">
          <div className="view-tabs" role="tablist">
            {TABS.map((tab) => (
              <button
                key={tab.key}
                className="view-tab"
                role="tab"
                aria-selected={active === tab.key}
                onClick={() => setActive(tab.key)}
              >
                <span className="view-tab-title">{tab.title}</span>
                <span className="view-tab-desc">{tab.desc}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="panel-shell">
          <div className="panel-head">
            <div className="panel-head-left">
              <span className="dot" />
              <span>{active.toUpperCase()} · STANDBY</span>
            </div>
            <Clock />
          </div>
          <div className="panel-views">
            <div className="panel-view" data-active="true">
              <ActiveChart />
            </div>
          </div>
          <div className="panel-foot">
            <span>任務由 Hermes Agent 透過 API 送出</span>
            <span>此站僅呈現，不提供操作</span>
          </div>
        </div>
        </div>
      </div>
    </section>
  );
}
