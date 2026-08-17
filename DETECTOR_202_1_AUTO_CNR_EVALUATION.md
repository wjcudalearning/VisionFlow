# Detector 202-1 自動 CNR 演算法評估

- 日期：2026-08-17
- 評估對象：`detectors/detector_202_1.py`
- 比較基準：`detectors/detector_202.py`，一般二值化門檻 `172`
- 程式基準：VisionFlow commit `fda8b60a1491867c87889ed47a59b7aa5eb80b58`
- 參考來源：[`Wwjyun/AcceptanceChecker` 的 `DefectDetector`](https://github.com/Wwjyun/AcceptanceChecker/blob/117fce477744188b97659a035b031fe3bf874260/acceptance_checker/core/detector.py)，commit `117fce477744188b97659a035b031fe3bf874260`

## 技術結論

Detector 202-1 的主要優勢不是「完全不用門檻」，而是把固定灰階門檻改成依影像雜訊自動調整的局部異常門檻。它先估計緩慢變化的背景，再對背景扣除後的 residual 使用 MAD（Median Absolute Deviation，中位絕對偏差）估算雜訊，因此對整張影像一起變亮或變暗，比固定二值化 `gray > 172` 穩定很多。

不過，它仍有一個固定的絕對門檻下限 `8 DN`，而且 `3 × 3 Open` 會消除過細或過弱的訊號。所以「Auto CNR 可以承受任意光衰」並不成立；當缺陷 residual 衰減到約 8 個灰階以下，或雜訊相對缺陷變得太大時，仍會失效。

本次使用實際 Detector 202／202-1 程式做合成影像壓測。在「背景 150 DN、亮缺陷 220 DN、11 × 11 px」的案例中：

| 光衰與雜訊模型 | 固定門檻 172 | Auto CNR | 解讀 |
|---|---:|---:|---|
| 理想等比例變暗、無雜訊 | 約 `21%` | 約 `77%` | 理論能力上限，不代表相機實機 |
| 等比例變暗後加入固定 `σ = 2 DN` read noise | 保守約 `18%` | 保守約 `71%` | 低雜訊工業相機的樂觀情境 |
| Shot noise（`4 e⁻/DN`）＋`1 DN` read noise | 保守約 `10%` | 保守約 `40%` | 本報告較接近實務、但仍是合成模型的估計 |

因此，目前對 Detector 202-1 最合理的工程結論是：

> 合成模型支持約 `40%` 的光衰容忍能力；在尚未使用實際相機、鏡頭、材料與真實缺陷驗證前，建議只把 `30%` 當成暫定設計容忍值，不應直接把 `40%` 或 `71%` 寫成量產保證規格。

這個 `30%` 是保留模型誤差與實機差異後的工程建議，不是演算法的硬性數學常數。最終數字必須以實際資料做分層光衰驗證。

證據信心評級為 **「可供工程方向使用，但必須附帶限制」**：公式與合成試驗計算已核對，可用於選擇實機驗證範圍；因為缺少真實相機光衰資料，尚不足以核准量產光衰規格。

## 重要觀念：它是「先找異常，再計算 CNR」

雖然名稱是 Auto CNR，但目前 Detector 202-1 並不是直接使用 `CNR > 某門檻` 來決定候選。

實際流程是：

1. 以 residual 與自動雜訊門檻建立候選 mask。
2. 對 mask 做形態學與 connected components。
3. 候選形成後，才計算每個候選的 CNR。
4. CNR 用於排序與 metadata，不是目前的 NG 最低門檻。
5. 只要有任何 component 通過 residual、形態學、面積與邊界條件，就輸出 NG。

因此，這個演算法更精確的名稱是「以 robust residual 建立候選，並用局部背景估算 CNR」。CNR 很重要，但目前主要負責描述候選強度，而不是直接否決低 CNR 候選。

## 演算法邏輯與公式

### 1. 灰階影像

輸入 BGR 小圖先轉成單通道灰階：

$$
I(x,y) = \operatorname{Gray}(BGR(x,y))
$$

Detector 202-1 的 Gray 使用專案共用 `PreprocessPlan`；CPU 以 OpenCV 結果為正確性基準。後續 Auto CNR 分析目前在 CPU 執行。

### 2. 用大範圍 Gaussian blur 估計背景

背景 kernel 會依影像短邊調整：

$$
k_0 = \left\lfloor \frac{\min(H,W)}{40} \right\rfloor
$$

$$
k = \operatorname{odd}(\operatorname{clip}(k_0,31,151))
$$

其中 `odd()` 代表如果是偶數就加 1。背景估計為：

$$
B(x,y) = G_k * I(x,y)
$$

`G_k` 是大小為 `k × k` 的 Gaussian kernel。這一步保留局部小缺陷，但吸收較慢的照明梯度、陰影或材料背景變化。

### 3. 計算 residual

$$
R(x,y) = I(x,y) - B(x,y)
$$

亮缺陷通常得到正 residual，暗缺陷通常得到負 residual。後續使用絕對值，所以兩種極性都可形成候選；這是固定 `THRESH_BINARY` Detector 202 只能直接找亮訊號時沒有的能力。

### 4. 以 MAD robust 地估算雜訊

先取 residual 中位數：

$$
m_R = \operatorname{median}(R)
$$

再計算 MAD：

$$
MAD = \operatorname{median}(|R-m_R|)
$$

將 MAD 換算成近似 Gaussian 標準差：

$$
\hat{\sigma}_{noise} = \max(1.4826 \times MAD,10^{-6})
$$

`1.4826` 是 Gaussian 分布下把 MAD 對應到標準差的常用比例。使用 median 而不是直接對 residual 求 standard deviation，可降低少數強缺陷或離群值把雜訊估計拉高的影響。

### 5. 自動 residual 門檻

$$
T_R = \max(8,3\hat{\sigma}_{noise})
$$

初始候選 mask 為：

$$
M_0(x,y)=
\begin{cases}
255, & |R(x,y)-m_R| > T_R \\
0, & \text{otherwise}
\end{cases}
$$

這裡有兩個門檻來源：

- `3σ`：依每張影像的雜訊自動調整，雜訊越大，候選門檻越高。
- `8 DN`：避免影像非常平滑時門檻過低，把微小量化誤差或材料紋理都當成缺陷。

所以 Auto CNR 並非完全沒有固定參數；`8 DN` 是它面對強光衰時最重要的硬限制之一。

### 6. Morphology Open

使用 `3 × 3` 矩形 kernel 執行一次 Open：

$$
M_1 = \operatorname{dilate}(\operatorname{erode}(M_0,K_{3\times3}),K_{3\times3})
$$

Open 的目的為移除孤立雜點，但也代表：

- 小於約 `3 px` 寬的細線可能被消除。
- 光衰後只剩少數像素超過門檻時，即使中心像素仍很亮，也可能無法形成候選。
- 光衰容忍度同時受 residual 強度與候選幾何影響，不是只有 CNR 一個數字。

### 7. 套用中心與四邊屏蔽

Detector 202-1 沿用 Detector 202 的屏蔽語意：

- 中心屏蔽預設半寬 `100`、半高 `630`。
- 共同內縮 `0`。
- 左 `15`、右 `26`、上 `50`、下 `20`。

$$
M_2 = M_1 \land M_{include}
$$

被排除的像素：

- 不會形成 candidate。
- 不納入 candidate 的局部背景 ring。

如果真實缺陷落入屏蔽區，不論 CNR 多高、光衰多少，都會被刻意忽略。因此評估光衰容忍度時，必須先確認缺陷位置位於有效區域。

### 8. Connected components 與自動面積限制

使用 8-connectivity：

$$
C = \operatorname{ConnectedComponents}_8(M_2)
$$

面積下限與上限為：

$$
A_{min}=\max(5,\lfloor 10^{-6}HW \rfloor)
$$

$$
A_{max}=\lfloor 0.05HW \rfloor
$$

也就是至少 5 pixels，且不能超過整張小圖的 5%。此外，component bbox 太靠近影像最外側 1 pixel 時會被排除，以避免大塊邊緣不均被當成缺陷。

### 9. 局部背景 ring

對 component bbox 寬高 `w_c, h_c`，背景 ring 外擴距離為：

$$
p=\operatorname{clip}(1.5\max(w_c,h_c),8,50)
$$

缺陷灰階值取 component 本身的像素；背景值取外擴區域內、不屬於該 component、且未被屏蔽的像素。若有效背景少於 20 pixels，則退回使用整張有效區域作為背景。

### 10. CNR

$$
\mu_d = \operatorname{mean}(I\mid component)
$$

$$
\mu_b = \operatorname{mean}(I\mid background\ ring)
$$

$$
\sigma_b = \operatorname{std}(I\mid background\ ring)
$$

$$
\Delta = |\mu_d-\mu_b|
$$

$$
CNR = \frac{\Delta}{\max(\sigma_b,10^{-6})}
$$

候選依 CNR 由高到低輸出。CNR 越高，代表候選和局部背景相比越容易分離；但它不是缺陷真實性的機率，也不應解讀成 `confidence = 99%`。

## 固定二值化 Detector 202 的邏輯

Detector 202 先灰階，再以固定門檻 `172` 做一般二值化：

$$
M_{fixed}(x,y)=
\begin{cases}
255, & I(x,y)>172 \\
0, & I(x,y)\le172
\end{cases}
$$

之後套屏蔽、使用 `RETR_LIST` 找 contours，保留面積 `5～100 px²`、以 perimeter 的 `2%` 近似後剛好四個頂點的輪廓，不限制凹凸。

固定門檻法的判斷是絕對灰階判斷。假設亮缺陷原始平均值為 $\mu_d$，光衰比例為 $L$，保留亮度比例為：

$$
a=1-L
$$

忽略雜訊時，缺陷至少必須滿足：

$$
a\mu_d > 172
$$

所以理論最大光衰為：

$$
L_{max,fixed}<1-\frac{172}{\mu_d}
$$

本次案例 $\mu_d=220$：

$$
L_{max,fixed}<1-\frac{172}{220}=21.82\%
$$

實測無雜訊結果為可承受 `21%`、`22%` 時失效，與公式一致。加入雜訊後，部分缺陷像素會先掉到門檻以下，四邊形和面積也可能被破壞，因此實際可靠範圍縮小到約 `10～18%`。

## Auto CNR 為何比固定門檻更能容忍光衰

若整張影像按比例變暗：

$$
I_a=aI
$$

Gaussian 背景近似也按比例變暗：

$$
B_a\approx aB
$$

因此 residual 近似為：

$$
R_a=I_a-B_a\approx aR
$$

若雜訊也等比例縮放，MAD 與 sigma 同樣近似乘上 $a$：

$$
\hat\sigma_a\approx a\hat\sigma
$$

當 `3σ` 是主要門檻時，缺陷 residual 與門檻一起縮放，相對可分離性近似保持不變。固定 `172` 不會跟著影像變暗，所以很快失效。

但當光衰大到 `8 DN` 下限開始主導時：

$$
T_R=8
$$

若能撐過 `3 × 3 Open` 的核心 residual 基準為 $r_{core,0}$，粗略必要條件是：

$$
ar_{core,0}>8
$$

$$
L_{max,auto}\approx1-\frac{8}{r_{core,0}}
$$

這解釋了為什麼無雜訊合成案例可到約 `77%` 光衰，但不會無限延伸。

## 更接近相機的光衰與雜訊模型

真實相機不會只把所有灰階完美乘上一個係數。可用下式近似背景雜訊：

$$
\sigma_b(a)=\sqrt{a\sigma_{shot,0}^2+a^2\sigma_{FPN,0}^2+\sigma_{read}^2}
$$

其中：

- $\sigma_{shot,0}$：基準曝光的 photon shot noise。
- $\sigma_{FPN,0}$：會隨訊號縮放的固定圖樣或空間不均。
- $\sigma_{read}$：不隨光量同比下降的 read noise／黑階雜訊。

缺陷對背景的原始灰階差為 $\Delta_0$ 時：

$$
CNR(a)\approx\frac{a\Delta_0}{\sqrt{a\sigma_{shot,0}^2+a^2\sigma_{FPN,0}^2+\sigma_{read}^2}}
$$

因此：

- 只有比例型雜訊時，contrast 與 noise 同時下降，CNR 可能大致維持。
- Shot noise 主導時，CNR 約隨 $\sqrt{a}$ 下降。
- Read noise 主導時，contrast 持續下降但 noise 不再下降，CNR 近似隨 $a$ 下降。
- 接近黑階時，量化、黑階 offset 與 clipping 會讓簡單比例模型失效。

Auto CNR 的實際失效點由兩個條件共同決定：

1. residual 是否高於 $\max(8,3\hat\sigma)$ 並形成至少 3 pixels 寬的 component。
2. component 是否通過面積、邊界與屏蔽條件。

目前沒有額外的 `CNR_min` 判定，因此 CNR 值本身不是第三道 gate。

## 合成光衰測試設計

### 測試影像

| 項目 | 設定 |
|---|---|
| 尺寸 | `240 × 160 px` |
| 背景 | `150 DN` |
| 亮缺陷 | `220 DN` |
| 缺陷尺寸 | `11 × 11 px` |
| Detector 202 | threshold `172`、非反相、面積 `5～100` |
| Detector 202-1 | AcceptanceChecker Auto CNR 邏輯 |
| 屏蔽 | 關閉，以隔離演算法光衰能力 |
| 命中定義 | 輸出 bbox 必須覆蓋缺陷中心；不是只要任意 NG |

光衰定義為：

$$
I_{signal,aged}=aI_{signal,baseline},\quad L=1-a
$$

### 三種雜訊情境

1. **理想無雜訊**：訊號乘上 $a$ 後直接量化。
2. **固定 read noise**：訊號乘上 $a$ 後加入 Gaussian `σ = 2 DN`。
3. **Shot＋read noise**：假設 `4 e⁻/DN`，先在電子域取 Poisson sample，再加入 Gaussian `1 DN` read noise。

第 2、3 種模型先以 50 seeds 掃描 `0～95%` 光衰；轉折區再以每個光衰點 300 seeds 重測，並計算 Wilson 95% confidence interval。粗掃 target seed 使用 `100000 × 光衰百分比 + seed`，空白背景 seed 再加 `9000000`；轉折區 seed 使用 `70000000 + 1000 × 光衰百分比 + seed`，確保同一設定可重建相同 sample。

## 光衰測試結果

### 理想無雜訊上限

| 光衰 | 固定門檻命中率 | Auto CNR 命中率 |
|---:|---:|---:|
| `0%` | `100%` | `100%` |
| `20%` | `100%` | `100%` |
| `30%` | `0%` | `100%` |
| `50%` | `0%` | `100%` |
| `70%` | `0%` | `100%` |
| `80%` | `0%` | `0%` |

逐 1% 掃描得到固定門檻最後成功點為 `21%`，Auto CNR 最後成功點為 `77%`。這是理想數學上限，不可直接作為量產規格。

### 固定 `2 DN` read noise

| 光衰 | 固定門檻命中率 | Auto CNR 命中率 | 每點樣本數 |
|---:|---:|---:|---:|
| `18%` | `100.0%` | `100.0%` | 300 |
| `19%` | `92.7%` | `100.0%` | 300 |
| `20%` | `20.3%` | `100.0%` | 300 |
| `70%` | `0%` | `99.7%` | 300 |
| `71%` | `0%` | `99.0%` | 300 |
| `72%` | `0%` | `88.7%` | 300 |
| `74%` | `0%` | `52.3%` | 300 |

以「命中率至少 95%」作為本次可靠命中定義，固定門檻的保守結果約 `18%`，Auto CNR 約 `71%`。這個模型的雜訊很低，因此對 Auto CNR 偏樂觀。

### Shot noise＋read noise

| 光衰 | 固定門檻命中率 | Auto CNR 命中率 | Auto CNR Wilson 95% CI | 每點樣本數 |
|---:|---:|---:|---:|---:|
| `10%` | `99.3%` | `100.0%` | `98.7～100.0%` | 300 |
| `12%` | `96.3%` | `100.0%` | `98.7～100.0%` | 300 |
| `13%` | `87.7%` | `100.0%` | `98.7～100.0%` | 300 |
| `38%` | `0%` | `99.3%` | `97.6～99.8%` | 300 |
| `40%` | `0%` | `99.3%` | `97.6～99.8%` | 300 |
| `41%` | `0%` | `95.7%` | `92.7～97.5%` | 300 |
| `44%` | `0%` | `96.0%` | `93.1～97.7%` | 300 |
| `45%` | `0%` | `91.3%` | `87.6～94.0%` | 300 |
| `50%` | `0%` | `78.3%` | `73.3～82.6%` | 300 |
| `52%` | `0%` | `58.0%` | `52.4～63.5%` | 300 |

因 Monte Carlo sampling，個別相鄰點可能小幅非單調，例如 `44%` 的點估計高於 `43%`。若要求點估計至少 95%，轉折約落在 `41～44%`；若再要求 Wilson 95% CI 下限也高於 95%，本次可保守支持到 `40%`。

固定門檻在 `12%` 的點估計仍為 96.3%，但其 Wilson CI 下限為 93.6%；最後一個 Wilson 95% CI 下限仍高於 95% 的點為 `11%`，摘要將它保守取整為約 `10%`。這說明固定門檻在臨界灰階附近會快速崩落，而 Auto CNR 的轉折較晚、較緩。

### 空白影像誤報觀察

在前段 50-seed、逐 1% 掃描中，固定 read-noise 與 shot＋read-noise 模型各測試 4,800 張無缺陷背景影像；兩種 Detector 的觀察誤報皆為 `0/4,800`。

這只能證明本次「均勻背景＋指定合成雜訊」沒有誤報，不能外推到具有材料紋理、刮痕、髒污、固定圖樣噪聲或局部反光的真實產品。

## 相較固定二值化的優點

| 面向 | 固定門檻 Detector 202 | Auto CNR Detector 202-1 |
|---|---|---|
| 全局亮度漂移 | 門檻 172 不變，容易因光衰漏檢 | 背景與 residual 門檻會隨影像調整，較穩定 |
| 局部照明梯度 | 可能把背景整片二值化 | Gaussian 背景扣除可抑制慢變化 |
| 缺陷極性 | 預設只直接接受亮於 172 的區域；可改反相，但單次設定不能同時處理兩種極性 | 使用 residual 絕對值，可同時找亮／暗異常 |
| 雜訊適應 | 無；同一門檻套所有影像 | MAD 依每張影像估 robust noise sigma |
| 幾何先驗 | 明確要求四邊形、面積 5～100 | 不要求四邊形，能找不規則 component |
| 可解釋性 | 非常直觀，灰階是否超過 172 | 仍可解釋，但步驟、統計量較多 |
| 運算量 | 低 | Gaussian、MAD、connected components 與 ring 統計較高 |
| 穩定環境的可驗證性 | 高，容易做硬性規格 | 需要驗證自動門檻、紋理與誤報 |
| 光衰監控能力 | 失效本身可能反映亮度不足 | 可能在光衰後仍正常抓缺陷，因此不能取代光源監控 |

Auto CNR 的價值是降低「同一缺陷只因整體亮度變化就跨過固定門檻」的敏感度。但固定門檻仍有合理用途：如果產線光學環境非常穩定、缺陷灰階規格明確，而且四邊形先驗能有效排除雜訊，固定法更簡單、更快，也更容易做硬性驗收。

## 限制與可能失效的情境

### 1. `8 DN` 仍是固定下限

光衰後 residual 低於 8 DN 時，即使背景非常乾淨，候選也不會產生。這是光衰容忍度的硬限制。

### 2. 全圖 MAD 可能受紋理或大量異常影響

材料紋理、週期條紋或大量缺陷可能提高 MAD，進而提高 `3σ` 門檻並漏掉較弱缺陷。MAD 比 standard deviation robust，但不是對任何污染比例都免疫。

### 3. 大 Gaussian 背景會影響缺陷尺度

- 小於背景 kernel 的局部缺陷通常會留下 residual。
- 很寬、很平緩的缺陷可能被當成背景吸收。
- 缺陷尺寸接近 kernel 時，residual 振幅與形狀會依尺度改變。

### 4. `3 × 3 Open` 不利細線

小裂痕、細刮傷或光衰後只剩 1～2 pixels 寬的訊號，可能在 Open 階段被移除。若產品重要缺陷包含細線，必須另做線狀缺陷資料集驗證。

### 5. 自動面積不是產品規格

`5 px` 到影像面積 `5%` 是通用 proxy，不一定等於產品允收尺寸。影像縮放、tile 尺寸改變或鏡頭倍率改變時，實體尺寸語意會跟著變。

### 6. 背景 ring 可能被鄰近異常污染

候選密集時，其他 component 的像素仍可能進入某一候選的背景統計，導致背景 mean/std 改變，CNR 下降或偏移。

### 7. 自動適應可能掩蓋光學系統退化

Detector 202-1 在光源衰退時仍可維持檢出，這對缺陷檢測是優點，但不能拿它證明照明系統健康。應另外監控背景灰階、曝光、Gain、均勻性和光源輸出。

### 8. 本報告不是實機驗收

合成模型未涵蓋：

- LED 光譜隨老化改變。
- 鏡頭污損與非均勻暗角。
- 相機 gamma、black level、auto exposure 或 auto gain。
- 材料反射率與偏振對光量的非線性反應。
- Bayer／ISP／壓縮處理。
- 真實 NG 尺度、形狀與位置分布。

因此，`40%` 是指定模型下的估計，不是跨設備通用常數。

## 如何在實機定義「光衰百分比」

不要只用 LED 控制器的設定百分比。應以扣除 black level 後的有效背景灰階量測：

$$
a=\frac{\widetilde I_{bg,aged}-I_{black}}{\widetilde I_{bg,baseline}-I_{black}}
$$

$$
L=1-a
$$

其中 $\widetilde I_{bg}$ 建議使用固定背景 ROI 的 median，避免少數缺陷或反光影響。相機 exposure、Gain、gamma、white balance 與 ISP 必須固定，否則量到的是相機補償後亮度，不是真正的光衰。

## 建議的實機驗證方案

### 第一階段：確認 30% 暫定容忍值

1. 固定 exposure、Gain、gamma、black level 與所有 ISP 設定。
2. 使用實際量測背景灰階定義 `0%、10%、20%、30%、40%、50%` 光衰。
3. 每個光衰階段至少包含：
   - 正常品。
   - 亮極性 NG。
   - 暗極性 NG。
   - 最小重要缺陷。
   - 不同 tile 位置，尤其靠近屏蔽邊界的位置。
4. 每一類至少收集多張獨立影像，不要只把同一張圖做數位乘法。
5. 同時執行 Detector 202 與 202-1，比較：
   - 目標定位 recall。
   - 每張圖誤報 component 數。
   - `robust_noise_sigma`。
   - `residual_threshold`。
   - 候選 CNR 分布。
   - component 面積與 bbox 穩定性。

### 第二階段：決定量產規格

量產可容忍光衰應定義為同時滿足下列條件的最大光衰點：

1. 所有關鍵缺陷類別的 recall 達到產品要求。
2. 正常品誤報率仍在產線可接受範圍。
3. 屏蔽邊界、component 面積與 bbox 沒有發生不可接受漂移。
4. 結果在不同相機、不同光源、冷熱機和不同批次材料上仍成立。
5. 光衰再增加一級時，系統能產生明確的照明維護警示，而不是靜默進入未知區域。

在取得這些實機證據前，建議採用：

- **演算法暫定設計容忍值：`30%` 光衰。**
- **合成模型保守能力估計：`40%` 光衰。**
- **超過 `30%`：要求維護警示或增加驗證，不直接視為已量產驗收。**

## 可考慮的後續改進

1. 新增獨立照明健康指標，避免 Auto CNR 掩蓋光衰。
2. 把 `CNR_min` 做成可選 Recipe gate，使 CNR 不只排序，也參與 NG 候選篩選。
3. 依產品實體尺寸取代目前通用的自動面積上限。
4. 對線狀缺陷提供不使用 `3 × 3 Open` 或使用方向性 morphology 的分支。
5. 背景 ring 排除所有其他 candidate，而不只排除目前 component。
6. 以實際資料建立光衰 × 缺陷類別 × 位置的 recall／false-positive 曲線。
7. 將背景 median、MAD、sigma、CNR 和光源使用時數做趨勢監控，提前安排維護。

## 尚待回答的問題

1. 實際小圖的基準背景灰階、重要缺陷灰階與 black level 各是多少？
2. 光衰主要是均勻亮度下降，還是伴隨暗角、色偏與局部不均？
3. 重要缺陷是否包含寬度小於 3 pixels 的裂痕或刮傷？
4. 可接受的 NG recall 與正常品誤報率規格是多少？
5. 是否需要將候選 CNR 納入最終 NG gate，而不是只用來排序？

回答以上問題後，才能把本報告的 `30%` 暫定設計值轉成正式量產規格。

## 證據與可重現性

- Auto CNR 參考公式：AcceptanceChecker commit `117fce477744188b97659a035b031fe3bf874260`。
- AOI 實作：`detectors/detector_202_1.py`。
- 固定二值化比較基準：`detectors/detector_202.py`。
- Auto CNR 等價與 routing 測試：`tests/test_detector_202_1.py`。
- 光衰試驗直接呼叫目前工作區的 Detector 202／202-1；合成影像在記憶體產生，未作為 production fixture 提交。
- 表格保留精確查閱用途；由於證據是受控合成試驗而非真實產線資料，本報告不以圖表暗示額外的統計代表性。
