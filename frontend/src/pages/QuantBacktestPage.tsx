import { useEffect, useRef, useState } from "react";
import {
  createChart,
  CandlestickSeries,
  createSeriesMarkers,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import "./QuantBacktestPage.css";

const API_BASE = "http://localhost:8000";

type Candle = { time: number; open: number; high: number; low: number; close: number };
type Trade = {
  direction: string;
  entry_time: number;
  entry_price: number;
  exit_time: number | null;
  exit_price: number | null;
  pnl: number;
  return_pct: number;
  status: string;
};
type Summary = {
  total_trades: number;
  win_rate_pct: number | null;
  total_return_pct: number;
  max_drawdown_pct: number;
  sharpe_ratio: number | null;
};
type DemoResponse = {
  candles: Candle[];
  trades: Trade[];
  summary: Summary;
  meta: { symbol: string; timeframe: string; start: string; end: string; bar_count: number };
};

function fmtTime(unixSec: number) {
  return new Date(unixSec * 1000).toLocaleString("zh-TW", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function QuantBacktestPage() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  const [status, setStatus] = useState<"loading" | "done" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [data, setData] = useState<DemoResponse | null>(null);

  useEffect(() => {
    setStatus("loading");
    fetch(`${API_BASE}/api/backtest/demo`)
      .then((res) => {
        if (!res.ok) throw new Error(`API 回應 ${res.status}`);
        return res.json();
      })
      .then((json: DemoResponse) => {
        setData(json);
        setStatus("done");
      })
      .catch((err) => {
        setErrorMsg(err.message || "計算失敗");
        setStatus("error");
      });
  }, []);

  useEffect(() => {
    if (!chartContainerRef.current || !data) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 480,
      layout: {
        background: { color: "transparent" },
        textColor: "#a8adb8",
      },
      grid: {
        vertLines: { color: "#23272d" },
        horzLines: { color: "#23272d" },
      },
      timeScale: { timeVisible: true },
    });
    chartRef.current = chart;

    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#3ecf8e",
      downColor: "#f0616b",
      borderVisible: false,
      wickUpColor: "#3ecf8e",
      wickDownColor: "#f0616b",
    });
    seriesRef.current = series;

    series.setData(
      data.candles.map((c) => ({
        time: c.time as Time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      }))
    );

    const markers: SeriesMarker<Time>[] = [];
    data.trades.forEach((t) => {
      markers.push({
        time: t.entry_time as Time,
        position: "belowBar",
        color: "#3ecf8e",
        shape: "arrowUp",
        text: "進",
      });
      if (t.exit_time) {
        markers.push({
          time: t.exit_time as Time,
          position: "aboveBar",
          color: "#f0616b",
          shape: "arrowDown",
          text: "出",
        });
      }
    });
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    createSeriesMarkers(series, markers);

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, [data]);

  return (
    <div className="quant-page">
      <h1>量化回測</h1>
      <p className="quant-page-sub">
        Demo：RSI 門檻策略（RSI 跌破 30 進場、突破 70 出場），TA-Lib 算指標、vectorbt 模擬，資料來自 quotes DB。
      </p>

      {status === "loading" && <div className="quant-status">計算中…</div>}
      {status === "error" && <div className="quant-status is-error">計算失敗：{errorMsg}</div>}

      {status === "done" && data && (
        <>
          <div className="quant-summary">
            <div className="quant-stat">
              <div className="quant-stat-label">Total Trades</div>
              <div className="quant-stat-value">{data.summary.total_trades}</div>
            </div>
            <div className="quant-stat">
              <div className="quant-stat-label">Win Rate</div>
              <div className="quant-stat-value">
                {data.summary.win_rate_pct?.toFixed(1) ?? "—"}%
              </div>
            </div>
            <div className="quant-stat">
              <div className="quant-stat-label">Total Return</div>
              <div
                className={
                  "quant-stat-value " +
                  (data.summary.total_return_pct >= 0 ? "is-positive" : "is-negative")
                }
              >
                {data.summary.total_return_pct.toFixed(2)}%
              </div>
            </div>
            <div className="quant-stat">
              <div className="quant-stat-label">Max Drawdown</div>
              <div className="quant-stat-value is-negative">
                {data.summary.max_drawdown_pct.toFixed(2)}%
              </div>
            </div>
            <div className="quant-stat">
              <div className="quant-stat-label">Sharpe</div>
              <div className="quant-stat-value">
                {data.summary.sharpe_ratio?.toFixed(2) ?? "—"}
              </div>
            </div>
          </div>

          <div className="quant-chart" ref={chartContainerRef} />

          <div className="quant-trades">
            <div className="quant-trades-wrap">
              <table>
                <thead>
                  <tr>
                    <th>方向</th>
                    <th>進場時間</th>
                    <th>進場價</th>
                    <th>出場時間</th>
                    <th>出場價</th>
                    <th>損益</th>
                    <th>報酬率</th>
                    <th>狀態</th>
                  </tr>
                </thead>
                <tbody>
                  {data.trades.map((t, i) => (
                    <tr key={i}>
                      <td>{t.direction}</td>
                      <td>{fmtTime(t.entry_time)}</td>
                      <td>{t.entry_price.toFixed(0)}</td>
                      <td>{t.exit_time ? fmtTime(t.exit_time) : "—"}</td>
                      <td>{t.exit_price ? t.exit_price.toFixed(0) : "—"}</td>
                      <td className={t.pnl >= 0 ? "is-positive" : "is-negative"}>
                        {t.pnl.toFixed(0)}
                      </td>
                      <td className={t.return_pct >= 0 ? "is-positive" : "is-negative"}>
                        {t.return_pct.toFixed(2)}%
                      </td>
                      <td>{t.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
