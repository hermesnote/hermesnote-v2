老大，可以。這份可以直接給 CLAUDE 作為設計討論依據。

**先講結論：**

1. **原始未來數值與模型學習目標分開是合理的，但「Label＝原始值、Target＝轉換後格式」不是通用標準命名。**
2. **波動度門檻、分位數門檻、交易成本門檻都有可用方式；嚴謹與否取決於時間對齊和估計程序，不是動態門檻一定比較好。**
3. **多 horizon 回歸預測再餵分類模型是合理的 stacking 方向，核心工程要求是產生「當時真正可取得的樣本外預測」。**
4. **Adam／AdamW 的原敘述需要反過來修正：傳統 Adam 的 coupled weight decay 通常就是加入 L2 梯度；AdamW 實作的是解耦 weight decay。**
5. **訓練介面除了模型超參數，更需要明確保存 horizon、lookback、資料切分、標籤規則及前處理版本。**

以下將**來源支持的概念**與**我對網站的設計建議**分開；搜尋範圍是建議起點，並非已在台指期驗證的最佳值。本次沒有執行實驗或修改文件。

---

# 第一塊：Label／Target 的定義與拆分

## 1. 目前拆法是否合理？

### 結論：運算拆分合理，命名可以更清楚

機器學習文獻中的 `label`、`target`、`response`、`y` 常有重疊：

- 分類的最終類別常稱 **label**。
- 分類或回歸的學習對象都可稱 **target**。
- 沒有普遍規則要求「Label 是原始連續數值、Target 是轉換後結果」。

我們自己的研究卡也直接把 Buy／Hold／Sell 稱為標籤，而不是只把報酬率稱為 Label。[1]

因此，我建議把目前概念改成以下幾個獨立設定：

| 設定 | 代表什麼 | 例子 |
|---|---|---|
| **預測對象 `outcome`** | 從未來資料定義的量 | 未來價格、簡單報酬率、對數報酬率 |
| **預測期間 `horizon`** | 往後看多遠 | 幾根 K 棒、多少交易時間 |
| **任務種類 `task_type`** | 要做什麼學習 | regression／classification |
| **目標生成規則 `target_transform` 或 `labeling_rule`** | 如何得到模型的 \(y\) | 原值、標準化、固定門檻、動態門檻 |
| **輸出編碼 `encoding`** | 如何交給特定模型／loss | 類別索引、浮點值；必要時 one-hot |

**重要差別：價格回歸與報酬率回歸，不只是「輸出格式不同」，而是預測對象不同。**

類別轉整數才比較接近純粹的「格式轉換」；把報酬率切成漲跌類別，是改變學習問題。

### 建議的三種組合

| 功能 | outcome | task_type | 生成 \(y\) 的方式 |
|---|---|---|---|
| 漲跌分類 | 未來報酬率 | classification | 依門檻切類別 |
| 未來價格回歸 | 未來價格 | regression | 原值或訓練集擬合的尺度轉換 |
| 未來報酬率回歸 | 未來報酬率 | regression | 原值或尺度轉換 |

這樣 Reproducer 可以忠實重建論文的目標，而 Experimenter 可以換模型，不必跟著重寫標籤定義。

---

## 2. 比命名更重要：時間與價格基準

假設資訊截至第 \(t\) 根 K 棒收盤才完整取得，常見 close-to-close 定義為：

\[
r_{t,h}=\frac{C_{t+h}}{C_t}-1
\]

或對數報酬率：

\[
\ell_{t,h}=\log\left(\frac{C_{t+h}}{C_t}\right)
\]

未來價格則是：

\[
y_{t,h}=C_{t+h}
\]

這些是**預測目標的定義，不自動等於可成交的交易報酬**。

如果訊號在 \(t\) 收盤後才產生、實際在下一根開盤進場，回測就要另外指定成交基準。可以：

- 預測 close-to-close 報酬，再由回測模擬真實成交；或
- 直接定義與可執行進出場對齊的 outcome。

兩者都可做，但要明確保存，不能混用。

### 網站需要明確記錄

- 資料頻率 `bar_interval`。
- 輸入歷史長度 `lookback`。
- 預測期間 `horizon_value`、`horizon_unit`。
- 資訊截止時間／K 棒時間戳代表開盤或收盤。
- 基準價格與終點價格欄位。
- 報酬率單位：小數比例或百分比。
- 夜盤、休市、缺棒、跨日與轉倉處理。
- 標籤何時完整可知：`label_end_time`／必要時更精確的 `label_available_time`。

**`bar_interval`、`lookback`、`horizon` 是三件不同的事，應獨立設定。**

另外，資料尾端尚未到期的樣本應標為「目標未知／不可用」，不能補成零報酬或「平」。

---

## 3. 分類 threshold：先把類別語意定準

### 二分類

真正的「漲／不漲」可以定義：

\[
y_{t,h}=
\begin{cases}
1,& r_{t,h}>0\\
0,& r_{t,h}\leq0
\end{cases}
\]

如果改成 \(r_{t,h}>\tau\)，正類的意思就變成：

> 「漲幅超過門檻」，負類包含小漲、持平與下跌。

因此，介面最好顯示規則，而不只是寫「漲／不漲」。

如果將中間區間樣本刪掉，只保留明顯漲與明顯跌，那是另一種 **有中性區排除規則的二分類**；必須記錄樣本排除比例，也要說明上線時如何處理所有時間點。

### 三分類

建議語意是 **跌／平／漲**，不是「漲／不漲／平」——後者會重疊。

令上下門檻都是正值：

\[
y_{t,h}=
\begin{cases}
\text{up},&r_{t,h}>\tau^+_{t,h}\\
\text{down},&r_{t,h}<-\tau^-_{t,h}\\
\text{flat},&\text{其他}
\end{cases}
\]

- 可以對稱，也可以不對稱。
- 邊界等號歸哪一類必須固定。
- 語意類別與模型編碼分開，例如語意是 down／flat／up，模型使用連續整數索引。

**標籤的報酬率門檻，和預測後「機率超過多少才交易」的決策門檻，也應分開。**

---

## 4. 是否有比固定百分比更嚴謹的方式？

有幾種不同目的的設計。

| 方式 | 定義方向 | 適用理由 | 需要注意 |
|---|---|---|---|
| **固定報酬率** | 固定上下界 | 易理解、可重現、適合 baseline | 不同波動環境／horizon 的類別比例可能差很多 |
| **波動度調整** | 門檻跟隨當下估計波動 | 將「顯著變動」改成相對市場噪音的尺度 | 波動估計必須只使用當時可得資料 |
| **分位數切分** | 用報酬分布的分位數設門檻 | 控制類別比例、做相對強弱分類 | 分位數只能從訓練／已成熟歷史標籤估計 |
| **成本導向** | 門檻包含預估交易成本 | 對齊可交易性 | 成本模型與方向、成交方式有關 |
| **Triple barrier** | 看期間內首次碰到上下界或時間界線 | 適合停利／停損／最長持有期的事件結果 | 這是路徑式標籤，不只是終點報酬切類別 |

### 波動度門檻的建議定義

\[
\tau^+_{t,h}=k^+\widehat{\sigma}_{t,h},
\qquad
\tau^-_{t,h}=k^-\widehat{\sigma}_{t,h}
\]

其中 \(\widehat{\sigma}_{t,h}\) 表示：**在 \(t\) 當時估計的、對應未來 \(h\) 期間的報酬波動尺度**。

可選估計方法：

- 歷史 rolling volatility。
- EWMA volatility。
- 由更完整的波動預測模型估計。

若用單根 K 棒波動乘上 \(\sqrt{h}\)，要標示這是依賴假設的近似；金融時間序列的自相關、盤中季節性與波動叢聚，可能讓近似失準。

如果用 ATR，則要處理單位：**ATR 是價格尺度，不能直接與報酬率比較**；需明確轉成相對價格尺度，且不能直接把 ATR 稱為報酬標準差。

López de Prado 的 *Advances in Financial Machine Learning* 第三章及其公開實作介面，提供以波動序列決定 barrier 寬度的具體先例。[3]

### 最容易漏掉的時間洩漏

假設使用「近期歷史 \(h\)-期報酬」估計分位數或波動：

- 某筆歷史報酬的起點雖然早於 \(t\)，
- 但它的終點若晚於 \(t\)，
- 在 \(t\) 當下仍然不可使用。

所以需要檢查的是**標籤成熟時間**，不只是資料列時間。

### 固定門檻不是不嚴謹

固定門檻只要預先定義、使用合理基準並做樣本外評估，就是有效 baseline。動態門檻增加適應性，也增加估計與調參的不確定性。

**我的建議：網站支援門檻方法擴充，但保留固定門檻作為可比較基準。**

### 研究卡中一個需釐清的地方

現有 multi-horizon 研究卡記錄以固定上下報酬門檻切 Buy／Hold／Sell，同時使用了「triple-barrier style」描述。[1]

**僅憑研究卡列出的終點報酬規則，不能認定它實作了真正 first-touch triple barrier。** 本次尚未回查 PDF 對應公式，因此不把該名稱當成已確認的方法事實，也不照搬到網站。

如果要做真正 triple barrier，還需定義同一根 OHLC K 棒同時碰上下界時的先後判定；可用更細資料或明確的保守規則處理。

---

# 第二部分：多 horizon 回歸 → 分類 stacking

## 1. 有沒有相對應的方法？

**有相近且成熟的方法家族：stacked generalization／stacking，以及用預測結果作為第二階段特徵的設計。**

你的構想可表示為：

```text
不同 horizon／不同資料頻率的回歸模型
                    ↓
          各模型的樣本外預測
                    ↓
         meta-classifier 分類模型
                    ↓
            最终分類／機率
```

現有 SEMP-TA 研究卡記錄 XGBoost、LightGBM、SVR 作 base learners，Ridge 作 meta-learner。[2]

但需要區分：

- 它支持「多模型預測組合作為第二層輸入」這個方向。
- **不能據此說它已證明你提出的「多 horizon 回歸 → 分類」完全相同設計。**
- 該研究卡本身也指出作者對分類與回歸指標的敘述混雜，不宜直接照抄其結果或驗證法。

此外，金融 **meta-labeling** 通常是第二模型判斷第一模型的交易是否值得執行；與你這種回歸預測融合有關聯，但不是同義詞。

---

## 2. 最重要的規則：meta 模型必須吃真正樣本外預測

錯誤方式：

```text
Base model 在整份訓練集學習
→ 對同一份訓練集預測
→ 將這些預測餵給 meta model
```

這會讓 meta 模型看到 base model 已經學過答案後產生的預測，與正式上線時的誤差分布不同。

scikit-learn 的 Stacking 文件也直接警告：使用 `prefit`，且 base model 與 stacking model 在同一資料上訓練，會有很高的過擬合風險。[4]

### 時間序列建議流程

1. 保留最外層、完全未參與設計的測試期間。
2. 在開發期間做 expanding／rolling walk-forward。
3. 每個時間切分中，base model 只用更早、且標籤已成熟的資料訓練。
4. 生成下一段期間的預測，保存為 out-of-time meta features。
5. 用累積的樣本外預測訓練 meta-classifier。
6. 到外層測試開始前，依既定規則重訓模型，再整條流程評估。

調參、標準化、特徵選擇與 early stopping，都必須放在對應的內層資料流程中。

### 特別注意 sklearn 的預設

- `StackingClassifier` 預設的 KFold／StratifiedKFold，**即使 `shuffle=False`，也不代表訓練資料永遠早於驗證資料**。
- 純 walk-forward 的最早一段通常只有訓練用途，無法產生樣本外預測；`cross_val_predict` 要求的 partition 形式可能無法直接接受這種切法。
- 因此，**不要假定把 `cv` 換成 `TimeSeriesSplit` 就已完成安全的時序 stacking**。
- 你的 base learners 又是回歸、meta learner 是分類，混合 estimator 類型也不應假定能直接套一般分類 stacking 包裝器。

**對網站而言，明確生成與保存 meta-feature 資料集，比直接套黑箱 stacking class 更容易重現與追查。**

---

## 3. 必須處理的設計問題

| 問題 | 處理方式 |
|---|---|
| **不同 horizon 的標籤重疊** | 依每筆 `label_end_time` 排除跨越驗證起點的訓練標籤；不能只按起點切資料 |
| **不同 K 棒頻率未對齊** | 在預測時刻只使用已收完的高週期 K 棒，保存資料可用時間 |
| **未來價格尺度不同** | 可轉成相對基準價格的隱含報酬，或採訓練集擬合的尺度轉換 |
| **不同 horizon 的預測意義不同** | 每個預測欄位保留模型、horizon、資料頻率與 target 定義 |
| **第二層究竟預測什麼** | meta-classifier 要有自己明確的目標、horizon 與標籤规则 |
| **base model 重訓後分布改變** | 評估整條重訓流程，追蹤 meta features 分布，不只評估單一模型 |
| **誤差傳遞** | 比較单模型、直接分類、簡單融合與 stacking；確認提升不是來自洩漏 |
| **是否真的需要 stacking** | 以同一外層樣本外期間比較效果、穩定性與成本 |

「誤差累積」在此更準確是**誤差傳遞和分布改變**：meta 模型可能學會修正，也可能放大錯誤，不能先假定一定改善或一定惡化。

建議預測產物至少保存：

`prediction_time`、`model_id`、`training_cutoff`、`horizon`、`target_version`、`fold_id`、`prediction_value`。

---

# 第三塊：LSTM 應暴露哪些參數？

先區分：

- **介面必須有明確設定／保存**，不等於每一個都要加入搜尋。
- 模型、資料、訓練程序、評估規則應分組，避免全部塞在「模型超參數」。

## 1. 必要與優先設定

除了你已列出的 units、layers、dropout、batch size、learning rate：

| 參數／設定 | 建議層級 | 為什麼需要 |
|---|---|---|
| **`lookback`／sequence length** | 必要、優先調整 | 決定模型看到多少歷史；與 horizon 不同 |
| **`max_epochs`** | 必要 | 明確限制訓練長度 |
| **Early stopping：monitor、patience、min_delta、還原最佳權重** | 必要 | 控制過擬合、保存真正使用的 checkpoint |
| **Optimizer 種類** | 必要 | Adam／AdamW 行為不同，影響重現 |
| **Weight decay／L2 的方法與係數** | 必要可設定 | 必須說明到底哪一種正則化 |
| **Gradient clipping：方法與上限** | 優先 | LSTM 訓練可能遇到梯度爆炸 |
| **Loss 與輸出 head** | 必要 | 二分類、三分類、價格／報酬回歸要匹配 |
| **輸入及 target scaling** | 必要的資料設定 | 影響神經網路優化與回歸 loss 尺度 |
| **類別權重／sample weighting** | 分類時優先可選 | 中性類可能占比很高 |
| **Seed、框架版本與執行設定** | 必要保存 | 比較與重現的基本條件 |

Gradient norm clipping 有 Pascanu、Mikolov、Bengio 對 RNN 梯度問題的研究依據。[5]

### 實務起始搜尋範圍——是設計建議，不是文獻最佳值

| 項目 | 起始候選 |
|---|---|
| `lookback` | 16、32、64、128 根；再依資料頻率及特徵調整 |
| Gradient norm clipping | 關閉、0.5、1、5 |
| AdamW weight decay | 0，以及 \(10^{-6}\)～\(10^{-2}\) 的對數尺度搜尋 |
| Learning rate | \(10^{-4}\)～\(3\times10^{-3}\)，對數尺度搜尋 |
| Early-stopping patience | 10～20 epochs 作初始候選，配合驗證曲線調整 |

Early stopping 的 `min_delta` 取決於監控指標及尺度，不適合固定一個值跨價格回歸、報酬回歸與分類共用。

---

## 2. 進階可選設定

| 項目 | 建議 |
|---|---|
| **LR scheduler** | 可選 `none`／`ReduceLROnPlateau`；再擴充 cosine、warmup |
| **Bidirectional** | 進階可選，先用單向 baseline |
| **Optimizer betas、epsilon** | 保存預設值，進階再開放調整 |
| **Huber loss／delta** | 回歸可選；適用於降低極端誤差的主導程度，但 delta 要對應尺度 |
| **Head hidden size／head dropout** | 與 LSTM 本體分開設定 |
| **Stateful／跨 batch hidden state** | 進階功能；要有嚴格序列與重置規則 |
| **Projection、attention、複雜 pooling** | 視為模型變體，後續擴充，不只是一串一般參數 |

### Bidirectional 不必然有未來洩漏

如果輸入窗口全部位於預測時刻之前，雙向 LSTM 只是正向及反向閱讀**已知歷史窗口**，不必然洩漏。

會出問題的是：

- 把窗口內较晚資訊拿來預測較早時刻的輸出。
- 或資料窗口本身已包含預測時刻之後的資訊。

因此不應寫「時間序列一律禁止 bidirectional」，而應以任務的資訊截止時間判斷。單向 baseline 較簡單，雙向是否值得保留，要實驗比較。

### Dropout 欄位要說清楚作用位置

PyTorch `nn.LSTM(dropout=...)` 的定義是**堆疊 recurrent layers 之間的 dropout，最後一層之外**；單層 LSTM 沒有這種層間 dropout 可施加。[6]

因此，介面若只有一個模糊的 `dropout`：

- 使用者設了值，可能以為作用於所有 LSTM 狀態；
- 實际卻沒有作用在他想的地方。

建議至少分清：

- `inter_layer_dropout`
- `head_dropout`
- 若實作支援，再提供 `input_dropout`／`recurrent_dropout`

不同框架的同名參數不一定等價，Reproducer 必須保存框架與實際位置。

---

# 第四塊：Adam／AdamW 與 L2 的正確區分

## 1. 原問題中的敘述需要修正

原敘述：

> Adam 的 weight_decay 跟真正的 L2 不等價，AdamW 才是正確實作 L2。

更準確應是：

> **對 Adam 這類自適應 optimizer，L2 penalty 與解耦 weight decay 不等價。傳統 Adam 的 `weight_decay` 常實作為 L2 penalty；AdamW 才把 weight decay 從梯度更新中解耦。**

對 L2 penalty：

\[
L_{\text{total}}(\theta)
=
L_{\text{data}}(\theta)
+
\frac{\lambda}{2}\|\theta\|_2^2
\]

梯度會變成：

\[
g_{\text{total}}=g_{\text{data}}+\lambda\theta
\]

Adam 會把這部分一併放進一阶、二阶動量估計。

AdamW 則把參數衰減與上述自適應梯度路徑分開處理。這是 Loshchilov 與 Hutter 的 *Decoupled Weight Decay Regularization* 核心區分。[7]

**AdamW 不是「Adam 才終於正確算 L2」，而是採用不同的正則化更新方式。**

---

## 2. 這個應用該選哪一個？

### 我的工程建議

- **新建 LSTM baseline：可用 AdamW 作預設候選。**
- **保留 Adam**，供論文重現與對照。
- 一併保留 `weight_decay=0`，不要假設正則化一定改善。
- 是否較好，仍由時間序列樣本外結果決定。

AdamW 原論文的主要實驗不是台指期分鐘資料，因此不能據此聲稱 AdamW 已被證明更適合這個市場。[7]

### 網站應保存的細節

- optimizer 名稱與版本。
- 使用 coupled L2 或 decoupled weight decay。
- 係數。
- 作用參數群組：是否包含 bias、normalization、recurrent weights。
- 是否另外在 loss 加 L2，避免無意間重複正則化。

目前 PyTorch 上游 Adam 已有 `decoupled_weight_decay` 選項，官方程式文件說明設為 True 時等價於 AdamW。[8]  
**你的部署版本是否有該參數尚未確認；不要只依 optimizer 名稱推定實際行為。**

---

# 第五塊：XGBoost 參數與搜尋範圍

## 1. 你列的四個是否值得優先暴露？

**值得。** 特別是金融技術指標常有相關性、樣本又有時間相依，控制模型複雜度比只增加樹數重要。

但我會再把 **`min_child_weight`** 放到優先欄位。

| 參數 | 作用 | 建議優先度 |
|---|---|---|
| `subsample` | 每輪使用部分訓練樣本 | 高 |
| `colsample_bytree` | 每棵樹使用部分特徵 | 高 |
| `reg_lambda` | 葉節點權重的 L2 正則化 | 高 |
| `reg_alpha` | 葉節點權重的 L1 正則化 | 中高 |
| `min_child_weight` | 控制子節點最低 Hessian 總量 | 高 |
| `gamma`／`min_split_loss` | 要求分裂帶來足夠 loss 改善 | 進階 |
| Early stopping | 依驗證表現選實際迭代數 | 必要 |
| `objective`、`eval_metric` | 決定任務與訓練監控 | 必要 |

官方文件明確將 `min_child_weight` 定義為 **Hessian 總和**；分類時不能把它直接解釋成「最少幾筆樣本」。[9]

另外，`reg_alpha` 不是一般線性模型式的「特徵選擇係數」；它正則化的是樹的葉權重。

---

## 2. 建議初始搜尋範圍

以下是**我建議的有限起始空間**，不是 XGBoost 官方針對金融市場推薦的最佳區間。

| 參數 | 初始範圍／候選 | 搜尋方式 |
|---|---|---|
| `n_estimators` | 300～2000 | 當作迭代上限，配合 early stopping |
| `max_depth` | 2～6 | 整數 |
| `learning_rate` | 0.01～0.1 | 對數尺度 |
| `subsample` | 0.6～1.0 | 連續或少量離散候選 |
| `colsample_bytree` | 0.5～1.0 | 同上 |
| `min_child_weight` | 1、3、5、10、20 | 離散；依樣本量再擴充 |
| `reg_alpha` | 0，加上 \(10^{-4}\)～10 | 0 獨立候選，其餘對數尺度 |
| `reg_lambda` | 0，加上 \(10^{-3}\)～100 | 同上，包含官方常用預設 1 |
| `gamma` | 0、0.01、0.1、1、5 | 進階候選，依 objective／尺度調整 |
| `early_stopping_rounds` | 30～100 | 保存設定與最佳 iteration |

說明：

- 樹深先保守，不代表台指期一定只需要淺樹；是控制第一輪搜尋成本與過擬合的起點。
- learning rate 越小，可能需要更多迭代；若最佳點一直貼著上限，應檢查是否是上限不足。
- `reg_alpha`、`reg_lambda`、`gamma` 的效果會受 objective、樣本權重與 target 尺度影響，不能把分類範圍毫無調整地搬到價格回歸。
- 特徵很少時，`colsample_bytree` 太低可能丟掉必要訊息。

**`subsample` 在訓練區間內抽樣，不等於把 train／validation／test 隨機切分。** 時間序列切分仍需獨立保證。

---

## 3. 類別不平衡與評估

- 二分類可提供 `scale_pos_weight`，以**訓練 fold** 的負／正類比例作候選，不一定要照比例固定。
- 三分類使用明確的 sample／class weighting 流程，不把二分類參數直接套過去。
- 類別權重可能改變預測機率的校準，若後續以機率決定交易，需另外驗證。

建議：

- 二分類：log loss、PR-AUC／ROC-AUC、F1 或 balanced accuracy，依任務選主指標。
- 三分類：multiclass log loss、macro-F1、各類 precision／recall。
- 回歸：MAE／RMSE，並與簡單 baseline 比較。
- 價格回歸至少比較「未來價格等於目前價格」；報酬回歸至少比較零報酬或訓練期基準。
- 最終交易表現另外用成本、滑價與可成交條件回測，不能只由 accuracy 推論。

---

# 給 CLAUDE 的設計收斂

我建議網站把設定分成六組，而不是把所有東西混成模型參數：

| 設定組 | 核心內容 |
|---|---|
| **資料** | dataset 版本、頻率、交易時段、時間戳與價格基準 |
| **Target 定義** | outcome、horizon、labeling rule、threshold、類別語意 |
| **特徵與輸入** | feature set 版本、lookback、前處理 |
| **模型** | LSTM／XGBoost 結構與參數 |
| **訓練與驗證** | optimizer、loss、early stopping、切分、標籤成熟／重疊處理 |
| **預測與實驗產物** | 模型 ID、設定版本、樣本外預測、指標、stacking 關聯 |

**對 Reproducer 而言，重點是忠實記錄原文定義與實作差異；對 Experimenter 而言，重點是換模型或特徵時，能保持同一個 target 與驗證規則。**

---

## 本次依據與證據範圍

**[1] 本地研究卡：Multi-horizon 分類**  
*A Meta-Adaptive Framework Combining Gated Attention Mechanisms and Feature-Wise Modulation for Multi-Horizon Stock Price Movement Prediction*，DOI：`10.1109/ACCESS.2026.3663386`。  
位置：`research/library/extracted/2026_A Meta-Adaptive Framework Combining Gated Attention Mechanisms and Feature-Wise Modulation for Multi-Horizon Stock Price Movement Prediction.md`，Target／Label 段。  
本次讀的是研究卡，未重新核對 PDF；其中 triple-barrier 用語保留疑問。

**[2] 本地研究卡：Stacking**  
*SEMP-TA: A Novel Stock Market Prediction Approach Based on Stacking Ensemble Machine Learning for Effective Trend Analysis*，DOI：`10.1109/ACCESS.2025.3586233`。  
位置：`research/library/extracted/2025_SEMP-TA_ A Novel Stock Market Prediction Approach Based on Stacking Ensemble Machine Learning for Effective Trend Analysis.md`。  
支持架構先例，不把普通 nested CV 自動視為已排除時間洩漏。

**[3] 波動度門檻與 triple barrier**  
López de Prado，*Advances in Financial Machine Learning*，Chapter 3。本次核對其公開實作／介面說明：  
[mlfinpy labeling 原始碼](https://github.com/baobach/mlfinpy/blob/main/mlfinpy/labeling/labeling.py)。  
這是方法與工程參考，未在本次執行其程式。

**[4] Stacking 的樣本外預測與 prefit 警告**  
[scikit-learn StackingClassifier 文件](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)。

**[5] RNN gradient clipping**  
Pascanu、Mikolov、Bengio，*On the difficulty of training Recurrent Neural Networks*：  
https://arxiv.org/abs/1211.5063

**[6] PyTorch LSTM 定義與 dropout 行為**  
[官方原始碼／內嵌文件](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/rnn.py)。

**[7] AdamW 原論文**  
Loshchilov、Hutter，*Decoupled Weight Decay Regularization*，ICLR 2019：  
https://arxiv.org/abs/1711.05101

**[8] PyTorch Adam 的 weight decay 行為**  
[官方 Adam 原始碼／內嵌文件](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py)。

**[9] XGBoost 官方參數定義**  
[官方 parameter.rst](https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst)。

PyTorch／XGBoost 文件網站本次回傳 403，因此改讀官方 GitHub 文件與原始碼；這些是上游版本資料，不代表網站部署版本已具備相同功能。