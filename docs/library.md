# Feature Library Classification

> TA-Lib 大類沿用官方分類；中間次類為研究用途與消融實驗補上的分類。`Custom` 保存 TA-Lib 未提供、由本研究自行定義的特徵。
> 2026-09-03 更新：TA-Lib 161 個函式全部補齊到對應次類底下，附一句話中文說明（來源：`backend/indicators/descriptions.py`，跟後端 `/api/indicators` 回傳的內容同步）。次類本身的樹狀結構不變，只是把示意用的「SMA、EMA」補成完整清單。

```text
Feature Library
│
├── TA-Lib
│   │
│   ├── Overlap Studies｜疊加型指標
│   │   ├── Moving Average｜均線
│   │   │   ├── SMA — 簡單移動平均，固定天期內收盤價的算術平均
│   │   │   ├── EMA — 指數移動平均，對近期價格給更高權重的移動平均線
│   │   │   ├── DEMA — 雙重指數移動平均，比一般 EMA 更快反應價格變化、減少延遲
│   │   │   ├── TEMA — 三重指數移動平均，比 DEMA 更進一步降低延遲
│   │   │   ├── TRIMA — 三角移動平均，對中間資料給較高權重的移動平均
│   │   │   ├── WMA — 加權移動平均，對近期資料線性給予較高權重
│   │   │   ├── KAMA — 考夫曼調適性移動平均，會依市場波動快慢自動調整反應速度
│   │   │   ├── MAMA — MESA 調適性移動平均，跟隨市場週期自動調整平滑程度
│   │   │   ├── MA — 移動平均（可指定簡單/指數等多種算法）
│   │   │   ├── MAVP — 可變週期移動平均，天期可以隨每個時間點指定不同數值
│   │   │   ├── T3 — 三重指數平滑移動平均，比 EMA/DEMA 更平滑、延遲更小
│   │   │   └── HT_TRENDLINE — 希爾伯特轉換瞬時趨勢線，用週期分析算出的平滑趨勢線
│   │   ├── Bands / Channels｜通道
│   │   │   ├── BBANDS — 布林通道，均線加減N倍標準差形成的上下通道，判斷超買超賣或突破
│   │   │   └── ACCBANDS — 加速通道，依價格波動幅度動態調整的上下通道
│   │   └── Stops / Overlays｜停損線／疊加線
│   │       ├── SAR — 拋物線停損轉向指標，沿價格走勢畫出的動態停損點，趨勢反轉時會翻面
│   │       ├── SAREXT — SAR 的進階版，可分別設定多空的加速因子
│   │       ├── MIDPOINT — 一段期間內最高價與最低價的中點
│   │       └── MIDPRICE — 一段期間內最高價與最低價的中點（用 high/low 序列）
│   │
│   ├── Momentum Indicators｜動能指標
│   │   ├── Strength Oscillator｜強弱震盪
│   │   │   ├── RSI — 相對強弱指標，衡量近期漲跌力道比例，判斷超買超賣
│   │   │   ├── STOCH — 隨機指標（KD），比較收盤價在近期高低區間中的相對位置
│   │   │   ├── STOCHF — 快速隨機指標，STOCH 未經平滑的版本，反應更靈敏
│   │   │   ├── STOCHRSI — 隨機 RSI，把 RSI 再套一次隨機指標算法，放大靈敏度
│   │   │   ├── WILLR — 威廉指標（跟 KD 相反方向），衡量收盤價在近期高低區間的相對位置
│   │   │   ├── CCI — 順勢指標，價格偏離平均值的程度，用來判斷超買超賣
│   │   │   ├── CMO — 錢德動量震盪指標，衡量近期漲跌力道的淨值
│   │   │   ├── MFI — 資金流量指標，結合價格與成交量的 RSI，衡量買賣壓力
│   │   │   ├── ULTOSC — 終極震盪指標，結合三種不同天期的動能，避免單一週期失真
│   │   │   ├── BOP — 力道平衡指標，用開高低收比較買賣力道強弱
│   │   │   └── IMI — 盤中動量指標，用開盤收盤價比較，類似 RSI 但只看當根K棒
│   │   ├── Directional Strength｜方向強度
│   │   │   ├── ADX — 平均方向指數，衡量趨勢強度（不分方向），數值越高趨勢越強
│   │   │   ├── ADXR — ADX 的平滑版本，比較目前與前期 ADX，用來確認趨勢強度是否穩定
│   │   │   ├── DX — 方向性指標，ADX 計算過程中的方向性數值
│   │   │   ├── MINUS_DI — 下降方向指標，ADX 系統中衡量空方力道的分量
│   │   │   ├── PLUS_DI — 上升方向指標，ADX 系統中衡量多方力道的分量
│   │   │   ├── MINUS_DM — 下降方向動量，MINUS_DI 計算過程中的原始動量值
│   │   │   ├── PLUS_DM — 上升方向動量，PLUS_DI 計算過程中的原始動量值
│   │   │   ├── AROON — 阿隆指標，衡量距離最近的最高價/最低價已經過了多久，判斷趨勢是否轉弱
│   │   │   └── AROONOSC — 阿隆震盪指標，AROON 上升線減下降線，單一數值判斷多空力道
│   │   └── Rate / Price Oscillator｜變化率／價格震盪
│   │       ├── MACD — 指數平滑異同移動平均，快慢均線差值搭配訊號線，判斷動能轉折
│   │       ├── MACDEXT — MACD 的進階版，可自訂各均線的計算方式
│   │       ├── MACDFIX — MACD 的固定參數版本（12/26/9 天期固定，只能調訊號線）
│   │       ├── APO — 絕對價格震盪指標，兩條不同天期均線的差（類似 MACD 但不含訊號線）
│   │       ├── PPO — 百分比價格震盪指標，MACD 的百分比版本，方便跨商品比較
│   │       ├── ROC — 變動率，價格相對 N 期前的漲跌百分比
│   │       ├── ROCP — 變動率（比例），ROC 的比例表示法
│   │       ├── ROCR — 變動率（比率），目前價格與 N 期前價格的比率
│   │       ├── ROCR100 — 變動率（比率x100），ROCR 乘以 100 的表示法
│   │       ├── MOM — 動量指標，目前價格與 N 期前價格的差
│   │       └── TRIX — 三重平滑指數平均變化率，過濾雜訊後的動能指標
│   │
│   ├── Volume Indicators｜量能指標
│   │   ├── Cumulative Volume-Price｜累積量價
│   │   │   ├── OBV — 能量潮，把成交量依漲跌方向累加，判斷量價是否同步
│   │   │   └── AD — 累積派發線，結合價格位置與成交量，判斷資金是流入還流出
│   │   └── Volume Oscillator｜量能震盪
│   │       └── ADOSC — 佳慶震盪指標，AD 線的快慢均線差，判斷資金流向的動能變化
│   │
│   ├── Volatility Indicators｜波動指標
│   │   ├── Range Volatility｜區間波動
│   │   │   ├── TRANGE — 真實區間，單根K棒的真實波動幅度（含跳空）
│   │   │   └── ATR — 平均真實區間，衡量價格波動幅度的常用指標
│   │   └── Normalized Volatility｜標準化波動
│   │       └── NATR — 標準化平均真實區間，ATR 除以收盤價的百分比版本，方便跨商品比較
│   │
│   ├── Price Transform｜價格轉換
│   │   ├── Average Price｜平均價格
│   │   │   ├── AVGPRICE — 平均價，(開+高+低+收)/4
│   │   │   ├── TYPPRICE — 典型價，(最高+最低+收盤)/3
│   │   │   └── AVGDEV — 平均絕對離差，價格偏離平均值的絕對值平均
│   │   └── Weighted / Median Price｜加權／中位價格
│   │       ├── MEDPRICE — 中位價，(最高+最低)/2
│   │       └── WCLPRICE — 加權收盤價，(最高+最低+收盤x2)/4，給收盤價較高權重
│   │
│   ├── Cycle Indicators｜週期指標
│   │   ├── Dominant Cycle｜主導週期
│   │   │   └── HT_DCPERIOD — 主導週期，用希爾伯特轉換估計目前價格波動的週期長度（幾根K棒一循環）
│   │   ├── Phase｜相位
│   │   │   ├── HT_DCPHASE — 主導週期相位，目前價格位在週期循環中的哪個階段
│   │   │   ├── HT_PHASOR — 相量分量，把週期拆成同相/正交兩個分量，用於判斷週期轉折
│   │   │   └── HT_SINE — 正弦波指標，用正弦曲線提前預示週期轉折點
│   │   └── Cycle State｜週期狀態
│   │       └── HT_TRENDMODE — 趨勢/循環模式判斷，回傳目前市場是「趨勢行情」還是「循環（盤整）行情」
│   │
│   ├── Pattern Recognition｜型態辨識
│   │   ├── Single-Candle Pattern｜單根 K 型態
│   │   │   ├── CDLDOJI — 十字線，開收盤價幾乎相同，代表多空拉鋸不決
│   │   │   ├── CDLDRAGONFLYDOJI — 蜻蜓十字，長下影線、開收盤在最高點的十字線，常見底部訊號
│   │   │   ├── CDLGRAVESTONEDOJI — 墓碑十字，長上影線、開收盤在最低點的十字線，常見頭部訊號
│   │   │   ├── CDLLONGLEGGEDDOJI — 長腳十字，上下影線都很長的十字線，代表多空劇烈拉鋸不決
│   │   │   ├── CDLRICKSHAWMAN — 黃包車夫，長上下影線、實體在中間的十字線，多空拉鋸
│   │   │   ├── CDLHAMMER — 鎚子線，下降趨勢底部出現的長下影線小實體K線，反轉訊號
│   │   │   ├── CDLINVERTEDHAMMER — 倒鎚子線，下降趨勢底部出現的長上影線小實體K線，反轉訊號
│   │   │   ├── CDLHANGINGMAN — 上吊線，上升趨勢頂部出現的長下影線小實體K線，反轉訊號
│   │   │   ├── CDLSHOOTINGSTAR — 流星線，上升趨勢頂部出現的長上影線小實體K線，反轉訊號
│   │   │   ├── CDLMARUBOZU — 光頭光腳，開盤=最高/最低、收盤=最高/最低的長實體K線，動能強烈
│   │   │   ├── CDLCLOSINGMARUBOZU — 收盤光頭光腳，收盤價等於當日最高或最低價的K線
│   │   │   ├── CDLSPINNINGTOP — 紡錘線，上下影線接近、實體小的K線，代表多空僵持
│   │   │   ├── CDLHIGHWAVE — 大浪線，上下影線都很長、實體很小的K線，代表多空劇烈拉鋸
│   │   │   ├── CDLLONGLINE — 長實體線，實體特別長的K線，代表當根動能強烈
│   │   │   ├── CDLSHORTLINE — 短實體線，實體特別短的K線，代表當根動能疲弱
│   │   │   ├── CDLBELTHOLD — 捉腰帶線，開盤即為當日最高/最低價的長實體K線，帶方向性
│   │   │   └── CDLTAKURI — 探水竿，長下影線的十字線（鎚子線的變化型），底部訊號
│   │   ├── Two-Candle Pattern｜雙根 K 型態
│   │   │   ├── CDLENGULFING — 吞噬型態，後一根K棒實體完全吞噬前一根，反轉訊號
│   │   │   ├── CDLHARAMI — 母子型態，小實體K棒被前一根大實體完全包住，暗示趨勢趨緩
│   │   │   ├── CDLHARAMICROSS — 十字母子，母子型態中第二根是十字線，訊號更強
│   │   │   ├── CDLDARKCLOUDCOVER — 烏雲罩頂，空頭反轉型態，紅K後黑K深入前一根實體
│   │   │   ├── CDLPIERCING — 貫穿型態，多頭反轉型態，紅K深入前一根黑K實體一半以上
│   │   │   ├── CDLCOUNTERATTACK — 反擊線，兩根相反顏色但收盤價相近的K線，多空拉鋸後可能反轉
│   │   │   ├── CDLDOJISTAR — 十字星，趨勢中出現跳空十字線，暗示動能猶豫
│   │   │   ├── CDLMATCHINGLOW — 相同低點，兩根收盤價相近的黑K，暗示下跌動能減弱
│   │   │   ├── CDLHOMINGPIGEON — 家鴿型態，跟母子型態類似但兩根都是同色K線，弱勢反轉訊號
│   │   │   ├── CDLINNECK — 頸內線，下降趨勢中黑K後紅K收在前一根收盤附近，趨勢延續訊號
│   │   │   ├── CDLONNECK — 頸上線，下降趨勢中黑K後紅K收在前一根最低點附近，趨勢延續訊號
│   │   │   ├── CDLTHRUSTING — 插入線，下降趨勢中黑K後紅K未過前根中點，弱勢趨勢延續
│   │   │   ├── CDLKICKING — 反沖型態，兩根反向的光頭光腳K棒中間跳空，強烈反轉
│   │   │   ├── CDLKICKINGBYLENGTH — 反沖型態（依長度判斷），KICKING 的變化版，用實體長度判斷主導方向
│   │   │   └── CDLSEPARATINGLINES — 分離線，跳空同向開盤的兩根K線，暗示趨勢延續
│   │   └── Multi-Candle Pattern｜多根 K 型態
│   │       ├── CDL2CROWS — 兩隻烏鴉，空頭反轉型態，上升趨勢後出現兩根陰線吞噬
│   │       ├── CDL3BLACKCROWS — 三隻烏鴉，空頭反轉型態，連續三根長黑K線
│   │       ├── CDL3INSIDE — 三內部上升/下降，母子型態後再確認一根同向K棒
│   │       ├── CDL3LINESTRIKE — 三線打擊，三根同向K棒後被一根大逆向K棒完全吞噬
│   │       ├── CDL3OUTSIDE — 三外部上升/下降，吞噬型態後再確認一根同向K棒
│   │       ├── CDL3STARSINSOUTH — 南方三星，下降趨勢末端的少見底部反轉型態
│   │       ├── CDL3WHITESOLDIERS — 三白兵，多頭反轉型態，連續三根長紅K線
│   │       ├── CDLABANDONEDBABY — 棄嬰型態，跳空十字星後反向跳空，強烈反轉訊號
│   │       ├── CDLADVANCEBLOCK — 前進阻檔，上升趨勢中三根紅K但力道漸弱，暗示上漲動能不足
│   │       ├── CDLBREAKAWAY — 突破缺口，五根K棒的反轉型態，跳空後緩步反轉再確認
│   │       ├── CDLCONCEALBABYSWALL — 藏嬰吞噬，下降趨勢中少見的底部反轉型態
│   │       ├── CDLEVENINGDOJISTAR — 黃昏十字星，頂部反轉三根型態，中間為跳空十字星
│   │       ├── CDLEVENINGSTAR — 黃昏之星，頂部反轉三根型態，中間為跳空小實體
│   │       ├── CDLGAPSIDESIDEWHITE — 向上/向下跳空並列陽線，跳空後出現相似的紅K線，暗示趨勢延續
│   │       ├── CDLHIKKAKE — 陷阱型態，內困線後假突破再反轉的型態
│   │       ├── CDLHIKKAKEMOD — 修正版陷阱型態，HIKKAKE 的變化版本，多加了確認條件
│   │       ├── CDLIDENTICAL3CROWS — 三胞胎烏鴉，連續三根開盤約等於前一根收盤的長黑K，強烈空頭
│   │       ├── CDLLADDERBOTTOM — 梯底型態，下降趨勢末端連續黑K後出現反轉紅K
│   │       ├── CDLMATHOLD — 大量整理型態，上升趨勢中的短暫整理後延續型態
│   │       ├── CDLMORNINGDOJISTAR — 晨星十字，底部反轉三根型態，中間為跳空十字星
│   │       ├── CDLMORNINGSTAR — 晨星，底部反轉三根型態，中間為跳空小實體
│   │       ├── CDLRISEFALL3METHODS — 上升/下降三法，趨勢中出現逆勢小整理後延續原趨勢
│   │       ├── CDLSTALLEDPATTERN — 停頓型態，上升趨勢三紅K但最後一根明顯縮小，暗示動能停滯
│   │       ├── CDLSTICKSANDWICH — 條形三明治，兩根相同收盤價的黑K夾一根紅K，底部支撐訊號
│   │       ├── CDLTASUKIGAP — 跳空並列型態，跳空後同向K線被逆勢K線部分回補，趨勢延續
│   │       ├── CDLTRISTAR — 三星型態，連續三根十字線，強烈反轉訊號
│   │       ├── CDLUNIQUE3RIVER — 獨特三河床，底部反轉三根型態，少見但訊號明確
│   │       ├── CDLUPSIDEGAP2CROWS — 向上跳空兩隻烏鴉，上升趨勢中跳空後連兩根黑K，空頭反轉
│   │       └── CDLXSIDEGAP3METHODS — 跳空三法，跳空後三根K線確認趨勢延續方向
│   │
│   ├── Statistic Functions｜統計函式
│   │   ├── Dispersion｜分散程度
│   │   │   ├── STDDEV — 標準差，衡量價格在一段期間內的波動程度
│   │   │   └── VAR — 變異數，標準差的平方，衡量價格波動程度
│   │   ├── Correlation｜相關性
│   │   │   ├── CORREL — 皮爾森相關係數，衡量兩序列之間的線性相關程度
│   │   │   └── BETA — 貝他值，衡量標的相對另一序列（通常是大盤）的波動關聯性
│   │   └── Regression｜迴歸
│   │       ├── LINEARREG — 線性迴歸終值，用最近N期資料做線性迴歸後的預測值
│   │       ├── LINEARREG_ANGLE — 線性迴歸角度，迴歸線的傾斜角度，判斷趨勢陡峭程度
│   │       ├── LINEARREG_INTERCEPT — 線性迴歸截距，迴歸線的截距值
│   │       ├── LINEARREG_SLOPE — 線性迴歸斜率，迴歸線的斜率，代表趨勢方向與速度
│   │       └── TSF — 時間序列預測，用線性迴歸推算下一期的預測值
│   │
│   ├── Math Transform｜數學轉換
│   │   ├── Log / Root｜對數／根號（含進位/捨去）
│   │   │   ├── LN — 自然對數
│   │   │   ├── LOG10 — 以10為底的對數
│   │   │   ├── SQRT — 平方根
│   │   │   ├── EXP — 自然指數函數
│   │   │   ├── CEIL — 無條件進位取整
│   │   │   └── FLOOR — 無條件捨去取整
│   │   └── Trigonometric｜三角轉換
│   │       ├── SIN — 正弦函數
│   │       ├── COS — 餘弦函數
│   │       ├── TAN — 正切函數
│   │       ├── ASIN — 反正弦函數
│   │       ├── ACOS — 反餘弦函數
│   │       ├── ATAN — 反正切函數
│   │       ├── SINH — 雙曲正弦函數
│   │       ├── COSH — 雙曲餘弦函數
│   │       └── TANH — 雙曲正切函數
│   │
│   └── Math Operators｜數學運算
│       ├── Arithmetic｜基本運算
│       │   ├── ADD — 兩序列逐點相加
│       │   ├── SUB — 兩序列逐點相減
│       │   ├── MULT — 兩序列逐點相乘
│       │   └── DIV — 兩序列逐點相除
│       └── Rolling Operations｜滾動運算
│           ├── MAX — 一段期間內的最大值
│           ├── MIN — 一段期間內的最小值
│           ├── MAXINDEX — 一段期間內最大值出現的位置
│           ├── MININDEX — 一段期間內最小值出現的位置
│           ├── MINMAX — 同時回傳一段期間內的最大值與最小值
│           ├── MINMAXINDEX — 同時回傳最大值與最小值出現的位置
│           └── SUM — 一段期間內的加總
│
└── Custom
    │
    ├── Raw｜原始資料
    │   ├── Price｜價格
    │   │   └── Open、High、Low、Close — 原始開高低收價格，不做任何轉換的基礎資料
    │   ├── Volume｜成交量
    │   │   └── Volume — 原始成交量，不做任何轉換的基礎資料
    │   └── Candle Geometry｜K 棒幾何結構
    │       └── 實體比例、上影線比例、下影線比例 — K棒幾何特徵，各部位佔整根K棒的比例（尚未實作計算邏輯）
    │
    └── Structure｜價格結構
        ├── Key Levels｜關鍵價位
        │   └── 支撐、壓力、Pivot — 自動偵測近期支撐/壓力價位與樞紐點（尚未實作計算邏輯）
        ├── Events｜事件
        │   └── 突破、跌破、突破後回踩 — 價格突破/跌破關鍵價位、或突破後回踩測試的事件（尚未實作計算邏輯）
        └── States｜狀態
            └── 盤整、均線糾結、價格相對關鍵位 — 判斷目前是盤整、均線糾結、或價格相對關鍵位置的狀態（尚未實作計算邏輯）
```

## 對照表（161 個 TA-Lib 函式，來源與後端一致）

實際的參數 schema、`signal_supported`（有沒有接上進出場訊號模板）標記，以即時查詢為準：`GET /api/indicators`（後端 `indicators/registry.py` 產生）。這份文件的說明文字跟 `backend/indicators/descriptions.py` 同步維護，改一邊要記得改另一邊。
