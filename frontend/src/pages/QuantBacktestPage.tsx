import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createSeriesMarkers,
  type ISeriesApi,
  type LineData,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import "./QuantBacktestPage.css";

const API_BASE = import.meta.env.VITE_API_BASE as string;
const INITIAL_CANDLE_LIMIT = 1500;
const CHUNK_LIMIT = 1500;
const LOAD_MORE_THRESHOLD = 20; // 剩不到這麼多根就先去要下一段
const TRADES_PAGE_SIZE = 200;
const MIN_VISIBLE_BARS = 30; // 縮放最多只能放大到畫面裡至少還有這麼多根，跟 XQ 一樣，再細看不出什麼意義
const EDGE_MARGIN_BARS = 3; // 跳到最早/最新一筆時，留幾根空白，不要讓資料整個貼齊畫面邊緣
const TRADE_ZOOM_BARS = 100; // 點交易明細定位時，固定縮放到這個根數，不然縮太寬會被密集訊號淹沒

// lightweight-charts 把數字型的 time 當成 UTC 時間戳來排版座標軸，不會轉成瀏覽器本地時區，
// 導致圖表軸線時間跟下面交易明細表（用瀏覽器本地時間格式化）對不上，差了一個時區的量。
// 這裡把餵給圖表的時間統一加上本地時區位移，讓它「以為自己在顯示 UTC」時，顯示出來的
// 數字其實就是使用者的本地時間；從圖表拿回時間（十字準星、可視範圍）時要用 fromChartTime 換回真正的 unix time。
const TZ_OFFSET_SEC = -new Date().getTimezoneOffset() * 60;
function toChartTime(unixSec: number): Time {
  return (unixSec + TZ_OFFSET_SEC) as Time;
}
function fromChartTime(chartTime: number): number {
  return chartTime - TZ_OFFSET_SEC;
}

const MA_PERIODS = [5, 10, 20, 60, 100, 240] as const;
const MA_COLORS: Record<number, string> = {
  5: "#ffd54f",
  10: "#4fc3f7",
  20: "#81c784",
  60: "#ba68c8",
  100: "#ff8a65",
  240: "#f06292",
};

type Candle = { time: number; open: number; high: number; low: number; close: number; volume: number | null };
type Trade = {
  direction: string;
  entry_time: number;
  entry_price: number;
  entry_reason: string;
  exit_time: number | null;
  exit_price: number | null;
  exit_reason: string | null;
  points: number | null;
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
  profit_factor: number | null;
  avg_win_pct: number | null;
  avg_loss_pct: number | null;
  best_trade_pct: number | null;
  worst_trade_pct: number | null;
};
type Indicator = { key: string; label: string; params: Record<string, number | string> };
type Rule = { action: "entry" | "exit" | "filter"; text: string };
type JobDetail = {
  job_id: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  status: string;
  summary: Summary;
  indicators: Indicator[];
  rules: Rule[];
  bar_count: number;
  trade_count: number;
};

function fmtSigned(n: number | null | undefined, decimals: number, suffix = ""): string {
  if (n === null || n === undefined) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(decimals)}${suffix}`;
}

function fmtTime(unixSec: number) {
  return new Date(unixSec * 1000).toLocaleString("zh-TW", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Long 進場 / Short 出場（回補）是買進動作，Long 出場 / Short 進場是賣出動作
function actionLetter(direction: string, isEntry: boolean): "B" | "S" {
  const isLong = direction === "Long";
  if (isEntry) return isLong ? "B" : "S";
  return isLong ? "S" : "B";
}

// LB=開多 LS=平多 SS=開空 SB=平空——同樣是買進動作，L/S 前綴分辨這是多方還是空方
function actionLabel(direction: string, isEntry: boolean): string {
  const prefix = direction === "Long" ? "L" : "S";
  return prefix + actionLetter(direction, isEntry);
}

const SELECTED_MARKER_COLOR = "#e8b339";

type SelectedTrade = { entry_time: number; direction: string; exit_time: number | null };

function isSameTrade(t: Trade, sel: SelectedTrade | null): boolean {
  return !!sel && t.entry_time === sel.entry_time && t.direction === sel.direction && t.exit_time === sel.exit_time;
}

function buildMarkers(
  trades: Trade[],
  minTime: number,
  maxTime: number,
  selected: SelectedTrade | null
): SeriesMarker<Time>[] {
  const markers: SeriesMarker<Time>[] = [];
  for (const t of trades) {
    const isSelected = isSameTrade(t, selected);
    if (t.entry_time >= minTime && t.entry_time <= maxTime) {
      const letter = actionLetter(t.direction, true);
      markers.push({
        time: toChartTime(t.entry_time),
        position: letter === "B" ? "belowBar" : "aboveBar",
        color: isSelected ? SELECTED_MARKER_COLOR : letter === "B" ? "#f0616b" : "#3ecf8e",
        shape: letter === "B" ? "arrowUp" : "arrowDown",
        text: actionLabel(t.direction, true),
      });
    }
    if (t.exit_time && t.exit_time >= minTime && t.exit_time <= maxTime) {
      const letter = actionLetter(t.direction, false);
      markers.push({
        time: toChartTime(t.exit_time),
        position: letter === "B" ? "belowBar" : "aboveBar",
        color: isSelected ? SELECTED_MARKER_COLOR : letter === "B" ? "#f0616b" : "#3ecf8e",
        shape: letter === "B" ? "arrowUp" : "arrowDown",
        text: actionLabel(t.direction, false),
      });
    }
  }
  markers.sort((a, b) => (a.time as number) - (b.time as number));
  return markers;
}

export default function QuantBacktestPage() {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const jumpToTsRef = useRef<((targetTs: number) => void) | null>(null);
  const jumpToTradeRef = useRef<((entryTime: number) => void) | null>(null);
  const nudgeRef = useRef<((dir: -1 | 1) => void) | null>(null);
  const jumpToStartRef = useRef<(() => void) | null>(null);
  const jumpToEndRef = useRef<(() => void) | null>(null);
  const selectedTradeRef = useRef<SelectedTrade | null>(null);
  const [selectedTrade, setSelectedTrade] = useState<SelectedTrade | null>(null);
  const scrollTrackRef = useRef<HTMLDivElement>(null);
  const scrollThumbRef = useRef<HTMLDivElement>(null);
  const scrollBarCountRef = useRef<HTMLSpanElement>(null);
  const infoOhlcvRef = useRef<HTMLDivElement>(null);
  const infoMaRef = useRef<HTMLDivElement>(null);
  const [searchParams] = useSearchParams();

  const [status, setStatus] = useState<"loading" | "done" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [job, setJob] = useState<JobDetail | null>(null);

  const [tradeRows, setTradeRows] = useState<Trade[]>([]);
  const [tradeOffset, setTradeOffset] = useState(0);

  const [maEnabled, setMaEnabled] = useState<Record<number, boolean>>(() =>
    Object.fromEntries(MA_PERIODS.map((p) => [p, false]))
  );
  const maEnabledRef = useRef(maEnabled);
  const maSeriesRef = useRef<Map<number, ISeriesApi<"Line">> | null>(null);
  useEffect(() => {
    maEnabledRef.current = maEnabled;
    maSeriesRef.current?.forEach((s, p) => s.applyOptions({ visible: maEnabled[p] }));
  }, [maEnabled]);

  const jobIdParam = searchParams.get("job");

  useEffect(() => {
    setStatus("loading");

    async function resolveJobId(): Promise<string | null> {
      if (jobIdParam) return jobIdParam;
      const res = await fetch(`${API_BASE}/api/backtest/jobs?limit=10`);
      const list: { job_id: string; status: string }[] = await res.json();
      const done = list.find((j) => j.status === "done");
      return done ? done.job_id : null;
    }

    resolveJobId()
      .then((id) => {
        if (!id) throw new Error("尚無已完成的回測結果，請先到後台建立一次回測");
        return fetch(`${API_BASE}/api/backtest/jobs/${id}`);
      })
      .then((res) => {
        if (!res.ok) throw new Error(`API 回應 ${res.status}`);
        return res.json();
      })
      .then((json: JobDetail) => {
        if (json.status !== "done") throw new Error("這次回測還沒完成");
        setJob(json);
        setTradeOffset(0);
        setStatus("done");
      })
      .catch((err) => {
        setErrorMsg(err.message || "讀取失敗");
        setStatus("error");
      });
  }, [jobIdParam]);

  // 交易明細表分頁（交易數可能幾十萬筆，不能一次全部塞進表格）
  useEffect(() => {
    if (!job) return;
    fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/trades?offset=${tradeOffset}&limit=${TRADES_PAGE_SIZE}`)
      .then((res) => res.json())
      .then((json: { trades: Trade[] }) => setTradeRows(json.trades))
      .catch(() => setTradeRows([]));
  }, [job, tradeOffset]);

  useEffect(() => {
    const container = chartContainerRef.current;
    if (!container || !job) return;

    const chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: { background: { color: "transparent" }, textColor: "#a8adb8" },
      grid: {
        vertLines: { color: "#23272d" },
        horzLines: { color: "#23272d" },
      },
      timeScale: { timeVisible: true, maxBarSpacing: container.clientWidth / MIN_VISIBLE_BARS },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#f0616b",
      downColor: "#3ecf8e",
      borderVisible: false,
      wickUpColor: "#f0616b",
      wickDownColor: "#3ecf8e",
    });

    // 成交量副圖：獨立的 pane，放在主圖下方，用 stretch factor 讓它大約佔整體高度的 1/4
    const volumeSeries = chart.addSeries(
      HistogramSeries,
      { color: "#5b6270", priceFormat: { type: "volume" } },
      1
    );
    chart.panes()[0]?.setStretchFactor(3);
    chart.panes()[1]?.setStretchFactor(1);

    // MA 疊圖：六條都先建立，用 visible 開關而不是重建，切換的時候才不用整個圖表重畫
    const maSeriesMap = new Map<number, ISeriesApi<"Line">>();
    for (const period of MA_PERIODS) {
      const s = chart.addSeries(LineSeries, {
        color: MA_COLORS[period],
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        visible: maEnabledRef.current[period],
      });
      maSeriesMap.set(period, s);
    }
    maSeriesRef.current = maSeriesMap;

    let loadedCandles: Candle[] = [];
    let loadingMore = false;
    let allLoaded = false;
    let suppressAutoLoad = false; // 跳轉期間先關掉，不讓「滑到左邊界自動補資料」跟跳轉的定位打架
    const maValueMaps = new Map<number, Map<number, number>>();

    function computeMA(period: number): LineData[] {
      const out: LineData[] = [];
      const valueMap = new Map<number, number>();
      let sum = 0;
      for (let i = 0; i < loadedCandles.length; i++) {
        sum += loadedCandles[i].close;
        if (i >= period) sum -= loadedCandles[i - period].close;
        if (i >= period - 1) {
          const value = sum / period;
          out.push({ time: toChartTime(loadedCandles[i].time), value });
          valueMap.set(loadedCandles[i].time, value);
        }
      }
      maValueMaps.set(period, valueMap);
      return out;
    }

    function updateInfoBar(hoverTime: number | undefined) {
      const ohlcvEl = infoOhlcvRef.current;
      const maEl = infoMaRef.current;
      if (!ohlcvEl || !maEl) return;
      let c: Candle | undefined;
      if (hoverTime !== undefined) c = loadedCandles.find((x) => x.time === hoverTime);
      if (!c) c = loadedCandles[loadedCandles.length - 1];
      if (!c) {
        ohlcvEl.textContent = "";
        maEl.innerHTML = "";
        return;
      }
      const up = c.close >= c.open;
      ohlcvEl.className = "quant-chart-info-group " + (up ? "is-positive" : "is-negative");
      ohlcvEl.textContent =
        `開 ${c.open.toFixed(0)}  高 ${c.high.toFixed(0)}  低 ${c.low.toFixed(0)}  收 ${c.close.toFixed(0)}` +
        `  量 ${c.volume != null ? c.volume.toFixed(0) : "—"}`;
      maEl.innerHTML = MA_PERIODS.filter((p) => maEnabledRef.current[p])
        .map((p) => {
          const val = maValueMaps.get(p)?.get(c!.time);
          if (val === undefined) return "";
          return `<span style="color:${MA_COLORS[p]}">MA${p} ${val.toFixed(0)}</span>`;
        })
        .filter(Boolean)
        .join("  ");
    }

    let markersRequestId = 0;

    function applyData() {
      series.setData(
        loadedCandles.map((c) => ({ time: toChartTime(c.time), open: c.open, high: c.high, low: c.low, close: c.close }))
      );
      volumeSeries.setData(
        loadedCandles.map((c) => ({
          time: toChartTime(c.time),
          value: c.volume ?? 0,
          color: c.close >= c.open ? "#f0616b" : "#3ecf8e",
        }))
      );
      for (const period of MA_PERIODS) {
        maSeriesMap.get(period)?.setData(computeMA(period));
      }
      updateInfoBar(undefined);
      if (loadedCandles.length > 0 && job) {
        const minTime = loadedCandles[0].time;
        const maxTime = loadedCandles[loadedCandles.length - 1].time;
        // 只抓目前圖表看得到這段時間範圍內的交易來畫標記，交易總數不管多大都跟這裡無關。
        // 縮放/平移常常連續觸發好幾次 applyData()，用一個遞增的 request id 擋掉舊的回應——
        // 不然舊的請求後回來，會把新的（正確的）標記蓋掉，變成畫面上一整串交易憑空消失。
        const requestId = ++markersRequestId;
        fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/trades/range?start=${minTime}&end=${maxTime}`)
          .then((res) => res.json())
          .then((trades: Trade[]) => {
            if (requestId !== markersRequestId) return; // 有更新的請求了，這筆回應已經過期
            createSeriesMarkers(series, buildMarkers(trades, minTime, maxTime, selectedTradeRef.current));
          })
          .catch(() => {});
      }
    }

    chart.subscribeCrosshairMove((param) => {
      updateInfoBar(param.time !== undefined ? fromChartTime(param.time as number) : undefined);
    });

    fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?limit=${INITIAL_CANDLE_LIMIT}`)
      .then((res) => res.json())
      .then((candles: Candle[]) => {
        loadedCandles = candles;
        if (candles.length < INITIAL_CANDLE_LIMIT) allLoaded = true;
        allLoadedRight = true; // 沒給 before/around，抓到的就是最新一段，右邊沒有更多了
        applyData();
        chart.timeScale().fitContent();
      });

    let loadingMoreRight = false;
    let allLoadedRight = false;

    function onVisibleRangeChange(range: { from: number; to: number } | null) {
      if (!range || loadedCandles.length === 0 || suppressAutoLoad) return;

      // 左邊界：往回補資料（往左拖／縮放到接近起點）
      if (!loadingMore && !allLoaded && range.from <= LOAD_MORE_THRESHOLD) {
        loadingMore = true;
        const before = loadedCandles[0].time;
        fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?before=${before}&limit=${CHUNK_LIMIT}`)
          .then((res) => res.json())
          .then((more: Candle[]) => {
            if (more.length === 0) {
              allLoaded = true;
              return;
            }
            loadedCandles = [...more, ...loadedCandles];
            applyData();
            // 補進去的資料會讓所有 bar 的 index 往右挪，手動把可視範圍平移回原本看的位置，避免畫面跳掉
            chart.timeScale().setVisibleLogicalRange({ from: range.from + more.length, to: range.to + more.length });
          })
          .finally(() => {
            loadingMore = false;
          });
        return;
      }

      // 右邊界：往右微調／拖曳超過目前已載入資料的範圍時，往後補資料。
      // 這裡是之前「最右邊出現一整排 B/S 標記」的真正原因——不是資料集真的到底了，
      // 是目前這段已載入的 K 棒用完了，圖表把僅剩的真實資料擠壓在畫面邊緣顯示。
      if (!loadingMoreRight && !allLoadedRight && range.to >= loadedCandles.length - LOAD_MORE_THRESHOLD) {
        loadingMoreRight = true;
        const after = loadedCandles[loadedCandles.length - 1].time + 1;
        fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?around=${after}&limit=${CHUNK_LIMIT}`)
          .then((res) => res.json())
          .then((more: Candle[]) => {
            if (more.length === 0) {
              allLoadedRight = true;
              return;
            }
            loadedCandles = [...loadedCandles, ...more];
            applyData();
            // setData() 換資料後，lightweight-charts 會自動把可視範圍推向新的右邊界（跟真的走勢圖一樣自動追最新），
            // 這裡不要這個行為——補資料只是把資料補齊，畫面應該留在原本看的位置，不然會連環觸發、越補越遠。
            chart.timeScale().setVisibleLogicalRange({ from: range.from, to: range.to });
          })
          .finally(() => {
            loadingMoreRight = false;
          });
      }
    }

    chart.timeScale().subscribeVisibleLogicalRangeChange(onVisibleRangeChange);

    const startTs = Math.floor(new Date(job.start_date).getTime() / 1000);
    const endTs = Math.floor(new Date(job.end_date).getTime() / 1000);
    const totalSpan = Math.max(endTs - startTs, 1);

    // 跳到某個時間點附近，目標會變成畫面最左邊那一根。跳之前先記住目前縮放的可視根數，
    // 跳完之後套用同樣的根數，不要用 fitContent() 蓋掉使用者滑鼠滾輪調好的縮放程度。
    function performJumpToTs(targetTs: number) {
      const prevRange = chart.timeScale().getVisibleLogicalRange();
      const visibleBars = prevRange ? prevRange.to - prevRange.from : INITIAL_CANDLE_LIMIT;
      // 目前縮放的根數可能比預設一次抓的量還大（例如剛載入時 fitContent 顯示的根數），
      // 抓的量要跟著縮放放大，不然會被下面的 min() 硬夾回抓到的資料筆數，縮放就保不住。
      const fetchLimit = Math.max(INITIAL_CANDLE_LIMIT, Math.ceil(visibleBars) + 50);
      suppressAutoLoad = true;
      fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?around=${Math.round(targetTs)}&limit=${fetchLimit}`)
        .then((res) => res.json())
        .then((candles: Candle[]) => {
          // 目標超過資料實際的最後一根（結束日期不代表剛好有資料到那一刻）就會抓空——
          // 改成抓最後可用的一段，不要把畫面清空
          if (candles.length === 0) {
            return fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?before=${endTs + totalSpan}&limit=${fetchLimit}`)
              .then((res) => res.json());
          }
          return candles;
        })
        .then((candles: Candle[]) => {
          if (candles.length === 0) return; // 真的完全沒有資料才放棄
          loadedCandles = candles;
          allLoaded = false; // 跳到新位置，往回滑的邊界要重新判斷
          allLoadedRight = false;
          applyData();
          if (loadedCandles.length > 0) {
            chart.timeScale().setVisibleLogicalRange({
              from: 0,
              to: Math.min(visibleBars, loadedCandles.length - 1),
            });
          }
        })
        .finally(() => {
          // 晚一點才恢復，讓上面 setVisibleLogicalRange 觸發的事件先被忽略掉，
          // 不然「跳轉定位」跟「滑到左邊界自動補資料」會連續觸發，把縮放又蓋掉
          setTimeout(() => {
            suppressAutoLoad = false;
          }, 300);
        });
    }
    jumpToTsRef.current = performJumpToTs;

    // 點交易明細定位：不要沿用目前的縮放（可能很寬，密集訊號會找不到剛剛點的那一筆），
    // 固定縮放到 TRADE_ZOOM_BARS 根，並盡量把進場那根放在畫面正中間。
    // 前後各抓一半，兩批合併後進場那根的 index 就是前半那批的筆數，用這個定位置中；
    // 如果剛好在資料集頭尾、前後湊不滿，就往有資料的那邊挪，盡量維持根數而不是硬置中。
    function jumpToTrade(entryTime: number) {
      const half = Math.ceil(TRADE_ZOOM_BARS / 2);
      suppressAutoLoad = true;
      Promise.all([
        fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?before=${entryTime}&limit=${half + 20}`).then((r) => r.json()),
        fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?around=${entryTime}&limit=${TRADE_ZOOM_BARS + 20}`).then((r) => r.json()),
      ])
        .then(([before, after]: [Candle[], Candle[]]) => {
          if (after.length === 0) return;
          loadedCandles = [...before, ...after];
          allLoaded = before.length < half + 20;
          allLoadedRight = after.length < TRADE_ZOOM_BARS + 20;
          applyData();
          const centerIdx = before.length; // 進場那根在合併後陣列裡的 index
          let from = centerIdx - Math.floor(TRADE_ZOOM_BARS / 2);
          let to = from + TRADE_ZOOM_BARS;
          if (from < 0) {
            to -= from;
            from = 0;
          }
          const maxIdx = loadedCandles.length - 1;
          if (to > maxIdx) {
            from -= to - maxIdx;
            to = maxIdx;
          }
          chart.timeScale().setVisibleLogicalRange({ from: Math.max(from, 0), to });
        })
        .finally(() => {
          setTimeout(() => {
            suppressAutoLoad = false;
          }, 300);
        });
    }
    jumpToTradeRef.current = jumpToTrade;

    // 跳到最早/最新一筆：維持目前縮放的根數，資料那一端留 EDGE_MARGIN_BARS 根空白，
    // 不要讓第一根／最後一根貼死在畫面邊緣。
    function jumpToStart() {
      const prevRange = chart.timeScale().getVisibleLogicalRange();
      const visibleBars = prevRange ? prevRange.to - prevRange.from : INITIAL_CANDLE_LIMIT;
      const fetchLimit = Math.max(INITIAL_CANDLE_LIMIT, Math.ceil(visibleBars) + 50);
      suppressAutoLoad = true;
      fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?around=${startTs - totalSpan}&limit=${fetchLimit}`)
        .then((res) => res.json())
        .then((candles: Candle[]) => {
          if (candles.length === 0) return;
          loadedCandles = candles;
          allLoaded = true; // 已經是最早的資料，左邊界不用再補
          allLoadedRight = false;
          applyData();
          chart.timeScale().setVisibleLogicalRange({
            from: -EDGE_MARGIN_BARS,
            to: Math.min(visibleBars, loadedCandles.length - 1) - EDGE_MARGIN_BARS,
          });
        })
        .finally(() => {
          setTimeout(() => {
            suppressAutoLoad = false;
          }, 300);
        });
    }
    jumpToStartRef.current = jumpToStart;

    function jumpToEnd() {
      const prevRange = chart.timeScale().getVisibleLogicalRange();
      const visibleBars = prevRange ? prevRange.to - prevRange.from : INITIAL_CANDLE_LIMIT;
      const fetchLimit = Math.max(INITIAL_CANDLE_LIMIT, Math.ceil(visibleBars) + 50);
      suppressAutoLoad = true;
      fetch(`${API_BASE}/api/backtest/jobs/${job.job_id}/candles?limit=${fetchLimit}`)
        .then((res) => res.json())
        .then((candles: Candle[]) => {
          if (candles.length === 0) return;
          loadedCandles = candles;
          allLoaded = candles.length < fetchLimit;
          allLoadedRight = true; // 已經是最新的資料，右邊界不用再補
          applyData();
          const lastIdx = loadedCandles.length - 1;
          const to = lastIdx + EDGE_MARGIN_BARS;
          chart.timeScale().setVisibleLogicalRange({
            from: to - Math.min(visibleBars, loadedCandles.length - 1),
            to,
          });
        })
        .finally(() => {
          setTimeout(() => {
            suppressAutoLoad = false;
          }, 300);
        });
    }
    jumpToEndRef.current = jumpToEnd;

    // XQ 式滾動軸：軌道代表整個資料的時間範圍，thumb 的位置/寬度對應目前可視的時間範圍，
    // 每次可視範圍變化（縮放、平移、補資料）都重新算一次貼上去。
    function updateScrollbarThumb() {
      const track = scrollTrackRef.current;
      const thumb = scrollThumbRef.current;
      const barCountEl = scrollBarCountRef.current;
      if (!track || !thumb) return;
      const visibleRange = chart.timeScale().getVisibleRange();
      const logicalRange = chart.timeScale().getVisibleLogicalRange();
      if (!visibleRange) return;
      const startFrac = (fromChartTime(visibleRange.from as number) - startTs) / totalSpan;
      const endFrac = (fromChartTime(visibleRange.to as number) - startTs) / totalSpan;
      const leftPct = Math.max(0, Math.min(1, startFrac)) * 100;
      const widthPct = Math.max(Math.min(1, endFrac) - Math.max(0, startFrac), 0.005) * 100;
      thumb.style.left = `${leftPct}%`;
      thumb.style.width = `${widthPct}%`;
      if (barCountEl && logicalRange) {
        // 用 ceil(to)-floor(from) 而不是單純取寬度四捨五入——可視範圍的頭尾兩根常常只露出一部分，
        // 畫面上還是會整根畫出來，這樣算出來的根數才會跟你肉眼數的一致
        const rendered = Math.ceil(logicalRange.to) - Math.floor(logicalRange.from);
        barCountEl.textContent = String(Math.max(1, rendered));
      }
    }
    chart.timeScale().subscribeVisibleLogicalRangeChange(updateScrollbarThumb);

    // 拖曳 thumb：只平移（保持目前縮放的根數），跟 XQ 一樣，縮放交給滑鼠滾輪
    let dragStartX = 0;
    let dragStartLeftTs = 0;
    function onThumbDragMove(e: MouseEvent) {
      const track = scrollTrackRef.current;
      const thumb = scrollThumbRef.current;
      if (!track || !thumb) return;
      const deltaFrac = (e.clientX - dragStartX) / track.clientWidth;
      const newLeftTs = dragStartLeftTs + deltaFrac * totalSpan;
      const widthPct = parseFloat(thumb.style.width || "0");
      const leftPct = ((newLeftTs - startTs) / totalSpan) * 100;
      thumb.style.left = `${Math.max(0, Math.min(100 - widthPct, leftPct))}%`;
    }
    function onThumbDragEnd(e: MouseEvent) {
      window.removeEventListener("mousemove", onThumbDragMove);
      window.removeEventListener("mouseup", onThumbDragEnd);
      const track = scrollTrackRef.current;
      if (!track) return;
      const deltaFrac = (e.clientX - dragStartX) / track.clientWidth;
      const newLeftTs = dragStartLeftTs + deltaFrac * totalSpan;
      performJumpToTs(Math.max(startTs, Math.min(endTs, newLeftTs)));
    }
    function onThumbMouseDown(e: MouseEvent) {
      const visibleRange = chart.timeScale().getVisibleRange();
      dragStartX = e.clientX;
      dragStartLeftTs = visibleRange ? fromChartTime(visibleRange.from as number) : startTs;
      window.addEventListener("mousemove", onThumbDragMove);
      window.addEventListener("mouseup", onThumbDragEnd);
    }
    // 掛在整條軌道上，不是只掛在那個很細的 thumb 上——不用像素級精準才抓得到，
    // 在軌道上任何地方按住都可以拖（跟你截圖裡 XQ 的手感一樣）
    scrollTrackRef.current?.addEventListener("mousedown", onThumbMouseDown);

    // 左右微調箭頭：平移一根（tick），不改變縮放——縮放只靠滑鼠滾輪
    function nudge(dir: -1 | 1) {
      const range = chart.timeScale().getVisibleLogicalRange();
      if (!range) return;
      chart.timeScale().setVisibleLogicalRange({ from: range.from + dir, to: range.to + dir });
    }
    nudgeRef.current = nudge;

    const ro = new ResizeObserver(() => {
      chart.applyOptions({
        width: container.clientWidth,
        height: container.clientHeight,
        timeScale: { maxBarSpacing: container.clientWidth / MIN_VISIBLE_BARS },
      });
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      scrollTrackRef.current?.removeEventListener("mousedown", onThumbMouseDown);
      window.removeEventListener("mousemove", onThumbDragMove);
      window.removeEventListener("mouseup", onThumbDragEnd);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(updateScrollbarThumb);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(onVisibleRangeChange);
      jumpToTsRef.current = null;
      jumpToTradeRef.current = null;
      nudgeRef.current = null;
      jumpToStartRef.current = null;
      jumpToEndRef.current = null;
      maSeriesRef.current = null;
      chart.remove();
    };
  }, [job]);

  const tradePageStart = job ? Math.min(tradeOffset + 1, job.trade_count) : 0;
  const tradePageEnd = job ? Math.min(tradeOffset + TRADES_PAGE_SIZE, job.trade_count) : 0;

  return (
    <div className="quant-page">
      {status === "loading" && (
        <div className="quant-status quant-status-loading">
          回測中<span className="quant-status-dots"><span>.</span><span>.</span><span>.</span></span>
        </div>
      )}
      {status === "error" && <div className="quant-status is-error">{errorMsg}</div>}

      {status === "done" && job && (
        <div className="quant-body">
          <div className="quant-left">
            <div className="quant-left-section">
              <div className="quant-config-row-3">
                <div className="quant-config-box">{job.symbol}</div>
                <div className="quant-config-box">{job.timeframe}</div>
                <div className="quant-config-box quant-config-box-date">
                  {job.start_date}
                  <br />~ {job.end_date}
                </div>
              </div>
            </div>

            <div className="quant-left-section">
              <div className="quant-ma-bulk-btns">
                <button
                  type="button"
                  className="quant-ma-bulk-btn"
                  onClick={() => {
                    const allOn = MA_PERIODS.every((p) => maEnabled[p]);
                    setMaEnabled(Object.fromEntries(MA_PERIODS.map((p) => [p, !allOn])));
                  }}
                >
                  {MA_PERIODS.every((p) => maEnabled[p]) ? "全關" : "全開"}
                </button>
              </div>
              <div className="quant-ma-toggles">
                {MA_PERIODS.map((p) => (
                  <label key={p} className="quant-ma-toggle" style={{ color: MA_COLORS[p] }}>
                    <input
                      type="checkbox"
                      checked={maEnabled[p]}
                      onChange={(e) => setMaEnabled((prev) => ({ ...prev, [p]: e.target.checked }))}
                    />
                    MA{p}
                  </label>
                ))}
              </div>
            </div>

            <div className="quant-left-section">
              <div className="quant-chip-list is-scrollable">
                {job.indicators.map((ind, i) => (
                  <div className="quant-chip" key={i}>
                    {ind.label}
                    {Object.keys(ind.params).length > 0 && (
                      <span className="quant-chip-sub">
                        (
                        {Object.entries(ind.params)
                          .map(([k, v]) => `${k}=${v}`)
                          .join(", ")}
                        )
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="quant-left-section">
              <div className="quant-rule-list">
                {job.rules.map((r, i) => (
                  <div className="quant-rule" key={i}>
                    <span
                      className={
                        "quant-rule-tag " +
                        (r.action === "entry" ? "is-entry" : r.action === "exit" ? "is-exit" : "is-filter")
                      }
                    >
                      {r.action === "entry" ? "進" : r.action === "exit" ? "出" : "濾"}
                    </span>
                    <span className="quant-rule-text">{r.text}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="quant-left-section">
              <div className="quant-stat-grid">
                <div className="quant-stat">
                  <div className="quant-stat-label">交易數</div>
                  <div className="quant-stat-value">{job.summary.total_trades}</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">勝率</div>
                  <div className="quant-stat-value">{job.summary.win_rate_pct?.toFixed(1) ?? "—"}%</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">報酬</div>
                  <div
                    className={
                      "quant-stat-value " + (job.summary.total_return_pct >= 0 ? "is-positive" : "is-negative")
                    }
                  >
                    {fmtSigned(job.summary.total_return_pct, 2, "%")}
                  </div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">最大回撤</div>
                  <div className="quant-stat-value is-negative">{job.summary.max_drawdown_pct.toFixed(2)}%</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">夏普值</div>
                  <div className="quant-stat-value">{job.summary.sharpe_ratio?.toFixed(2) ?? "—"}</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">獲利因子</div>
                  <div className="quant-stat-value">{job.summary.profit_factor?.toFixed(2) ?? "—"}</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">平均獲利</div>
                  <div className="quant-stat-value is-positive">{fmtSigned(job.summary.avg_win_pct, 2, "%")}</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">平均虧損</div>
                  <div className="quant-stat-value is-negative">{fmtSigned(job.summary.avg_loss_pct, 2, "%")}</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">最佳單筆</div>
                  <div className="quant-stat-value is-positive">{fmtSigned(job.summary.best_trade_pct, 2, "%")}</div>
                </div>
                <div className="quant-stat">
                  <div className="quant-stat-label">最差單筆</div>
                  <div className="quant-stat-value is-negative">{fmtSigned(job.summary.worst_trade_pct, 2, "%")}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="quant-chart-wrap">
            <div className="quant-chart-plot">
              <div className="quant-chart" ref={chartContainerRef} />
              <div className="quant-chart-info-bar">
                <div className="quant-chart-info-group" ref={infoOhlcvRef} />
                <div className="quant-chart-info-group" ref={infoMaRef} />
              </div>
            </div>
            <div className="quant-chart-scrollbar">
              <button
                type="button"
                className="quant-scrollbar-arrow"
                onClick={() => jumpToStartRef.current?.()}
                aria-label="跳到最早一筆"
              >
                |&lt;
              </button>
              <span className="quant-scrollbar-date">{job.start_date}</span>
              <button
                type="button"
                className="quant-scrollbar-arrow"
                onClick={() => nudgeRef.current?.(-1)}
                aria-label="往左微調一根"
              >
                ◀
              </button>
              <div className="quant-scrollbar-track" ref={scrollTrackRef}>
                <div className="quant-scrollbar-thumb" ref={scrollThumbRef}>
                  <span ref={scrollBarCountRef} />
                </div>
              </div>
              <button
                type="button"
                className="quant-scrollbar-arrow"
                onClick={() => nudgeRef.current?.(1)}
                aria-label="往右微調一根"
              >
                ▶
              </button>
              <span className="quant-scrollbar-date">{job.end_date}</span>
              <button
                type="button"
                className="quant-scrollbar-arrow"
                onClick={() => jumpToEndRef.current?.()}
                aria-label="跳到最新一筆"
              >
                &gt;|
              </button>
            </div>
          </div>

          <div className="quant-trades">
            <div className="quant-trades-pager">
              <span>
                第 {tradePageStart}–{tradePageEnd} 筆，共 {job.trade_count.toLocaleString()} 筆
              </span>
              <div className="quant-trades-pager-btns">
                <button type="button" disabled={tradeOffset === 0} onClick={() => setTradeOffset(Math.max(0, tradeOffset - TRADES_PAGE_SIZE))}>
                  上一頁
                </button>
                <button
                  type="button"
                  disabled={tradeOffset + TRADES_PAGE_SIZE >= job.trade_count}
                  onClick={() => setTradeOffset(tradeOffset + TRADES_PAGE_SIZE)}
                >
                  下一頁
                </button>
              </div>
            </div>
            <div className="quant-trades-scroll">
            <table>
              <thead>
                <tr>
                  <th>方向</th>
                  <th>進場</th>
                  <th>進場時間</th>
                  <th>進場價</th>
                  <th>進場原因</th>
                  <th>出場</th>
                  <th>出場時間</th>
                  <th>出場價</th>
                  <th>出場原因</th>
                  <th>點數</th>
                  <th>損益</th>
                  <th>報酬率</th>
                  <th>狀態</th>
                </tr>
              </thead>
              <tbody>
                {tradeRows.map((t, i) => {
                  const entryLetter = actionLetter(t.direction, true);
                  const exitLetter = actionLetter(t.direction, false);
                  return (
                    <tr
                      key={i}
                      className={
                        "quant-trade-row" +
                        (isSameTrade(t, selectedTrade) ? " is-selected" : "")
                      }
                      onClick={() => {
                        const sel: SelectedTrade = { entry_time: t.entry_time, direction: t.direction, exit_time: t.exit_time };
                        selectedTradeRef.current = sel;
                        setSelectedTrade(sel);
                        jumpToTradeRef.current?.(t.entry_time);
                      }}
                      title="點擊定位到圖表上這筆交易的位置，並用金色標出這一筆"
                    >
                      <td>{t.direction}</td>
                      <td className={entryLetter === "B" ? "quant-mark-buy" : "quant-mark-sell"}>
                        {actionLabel(t.direction, true)}
                      </td>
                      <td>{fmtTime(t.entry_time)}</td>
                      <td>{t.entry_price.toFixed(0)}</td>
                      <td className="quant-reason-cell">{t.entry_reason}</td>
                      <td className={exitLetter === "B" ? "quant-mark-buy" : "quant-mark-sell"}>
                        {t.exit_time ? actionLabel(t.direction, false) : "—"}
                      </td>
                      <td>{t.exit_time ? fmtTime(t.exit_time) : "—"}</td>
                      <td>{t.exit_price ? t.exit_price.toFixed(0) : "—"}</td>
                      <td className="quant-reason-cell">{t.exit_reason ?? "—"}</td>
                      <td className={(t.points ?? 0) >= 0 ? "is-positive" : "is-negative"}>
                        {t.points !== null ? fmtSigned(t.points, 0) : "—"}
                      </td>
                      <td className={t.pnl >= 0 ? "is-positive" : "is-negative"}>{fmtSigned(t.pnl, 0)}</td>
                      <td className={t.return_pct >= 0 ? "is-positive" : "is-negative"}>{fmtSigned(t.return_pct, 2, "%")}</td>
                      <td>{t.status}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
