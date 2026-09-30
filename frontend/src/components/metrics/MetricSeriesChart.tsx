// 逐輪（或其他步驟序列）曲線：同一指標的 train／val 畫在同一張圖；標出 best 那一輪；
// 適用的基準畫成水平參考線。資料來自逐輪紀錄（REST 補歷史＋WebSocket 即時），圖表自己管理生命週期。
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart, createSeriesMarkers, LineSeries,
  type IChartApi, type ISeriesApi, type ISeriesMarkersPluginApi, type IPriceLine, type Time,
} from "lightweight-charts";
import { formatValue, pointValue, ROUND_NOUN, unitLabel, type MetricDef, type ProgressPoint, type SeriesSpec } from "./metricsModel";

const TRAIN_COLOR = "#f0616b";
const VAL_COLOR = "#3ab0cf";

export type RefLine = { label: string; value: number };

export default function MetricSeriesChart({
  series, def, points, roundUnit, headLabel, targetUnit, bestRound, refLines = [],
}: {
  series: SeriesSpec;
  def: MetricDef | undefined;
  points: ProgressPoint[];
  roundUnit: string;
  headLabel?: string;
  targetUnit?: string | null;
  bestRound?: number | null;
  refLines?: RefLine[];
}) {
  const boxRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const trainRef = useRef<ISeriesApi<"Line"> | null>(null);
  const valRef = useRef<ISeriesApi<"Line"> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const interactedRef = useRef(false);
  const [hover, setHover] = useState<number | null>(null);
  const noun = ROUND_NOUN[roundUnit] ?? roundUnit;

  const data = useMemo(() => {
    const byRound = new Map<number, ProgressPoint>();
    for (const p of points) byRound.set(p.epoch, p);
    const sorted = [...byRound.values()].sort((a, b) => a.epoch - b.epoch);
    const line = (key?: string) => (key ? sorted
      .map((p) => ({ time: (p.epoch + 1) as unknown as Time, value: pointValue(p, key) }))
      .filter((d): d is { time: Time; value: number } => d.value !== null) : []);
    return { sorted, train: line(series.train), val: line(series.val) };
  }, [points, series.train, series.val]);

  useEffect(() => {
    if (!boxRef.current) return;
    const priceFormatter = (v: number) => formatValue(def, v);
    const chart = createChart(boxRef.current, {
      layout: { background: { color: "transparent" }, textColor: "#a8adb8" },
      grid: { vertLines: { color: "#23272d" }, horzLines: { color: "#23272d" } },
      autoSize: true,
      localization: { timeFormatter: (t: number) => `第 ${t} ${noun}`, priceFormatter },
      timeScale: { timeVisible: false, tickMarkFormatter: (t: number) => String(t) },
    });
    const opts = { priceLineVisible: false, lastValueVisible: false,
      priceFormat: { type: "custom" as const, formatter: priceFormatter, minMove: 1e-9 } };
    trainRef.current = series.train ? chart.addSeries(LineSeries, { ...opts, color: TRAIN_COLOR }) : null;
    valRef.current = series.val ? chart.addSeries(LineSeries, { ...opts, color: VAL_COLOR, lineStyle: 2 }) : null;
    const anchor = valRef.current ?? trainRef.current;
    markersRef.current = anchor ? createSeriesMarkers(anchor, []) : null;
    chart.subscribeCrosshairMove((param) => setHover(param.time === undefined ? null : (param.time as number) - 1));
    const mark = () => { interactedRef.current = true; };
    const el = boxRef.current;
    el.addEventListener("wheel", mark, { passive: true });
    el.addEventListener("mousedown", mark);
    chartRef.current = chart;
    return () => {
      el.removeEventListener("wheel", mark);
      el.removeEventListener("mousedown", mark);
      chart.remove();
      chartRef.current = null;
      priceLinesRef.current = [];
    };
  }, [def, noun, series.train, series.val]);

  useEffect(() => {
    trainRef.current?.setData(data.train);
    valRef.current?.setData(data.val);
    if (!interactedRef.current) chartRef.current?.timeScale().fitContent();
  }, [data]);

  useEffect(() => {
    const anchor = valRef.current ?? trainRef.current;
    markersRef.current?.setMarkers(bestRound !== null && bestRound !== undefined && data.sorted.some((p) => p.epoch === bestRound)
      ? [{ time: (bestRound + 1) as unknown as Time, position: "aboveBar", color: "#e8b339", shape: "arrowDown", text: "best" }]
      : []);
    for (const pl of priceLinesRef.current) anchor?.removePriceLine(pl);
    priceLinesRef.current = anchor
      ? refLines.map((r) => anchor.createPriceLine({ price: r.value, color: "#6b7280", lineWidth: 1, lineStyle: 1, axisLabelVisible: true, title: r.label }))
      : [];
  }, [bestRound, refLines, data]);

  const shown = (hover !== null ? data.sorted.find((p) => p.epoch === hover) : undefined) ?? data.sorted[data.sorted.length - 1];
  const unit = unitLabel(series.unit ?? def?.unit, targetUnit);
  return (
    <div className="mv-chart" data-testid={`series-${series.id}`}>
      <div className="mv-chart-title">
        <strong>{def?.label ?? series.metric}</strong>
        {headLabel && <span className="mv-tag">{headLabel}</span>}
        {unit && <span className="mv-dim">{unit}</span>}
        {def?.direction && <span className="mv-dim">{def.direction === "max" ? "↑ 越高越好" : "↓ 越低越好"}</span>}
      </div>
      <div className="mv-legend">
        {series.train && <span><i className="mv-swatch" style={{ background: TRAIN_COLOR }} />train {formatValue(def, shown ? pointValue(shown, series.train) : null)}</span>}
        {series.val && <span><i className="mv-swatch is-dashed" style={{ borderColor: VAL_COLOR }} />val {formatValue(def, shown ? pointValue(shown, series.val) : null)}</span>}
        {shown && <span className="mv-dim">第 {shown.epoch + 1} {noun}</span>}
      </div>
      <div ref={boxRef} className="mv-chart-box" />
      {data.sorted.length === 0 && <div className="mv-empty">尚無逐輪資料</div>}
    </div>
  );
}
