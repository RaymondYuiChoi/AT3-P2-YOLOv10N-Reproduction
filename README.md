# AT3 · YOLOv10-N 雙重標籤分配復現

在 VOC2007、輸入 640、配對種子 2007、20 個 epoch 下比較 O2M、O2O 與 Dual。固定 E20 的官方 Val 上，Dual 的 AP50–95 高於 O2O 0.894 個百分點，低於 O2M 0.778 個百分點。O2M 沿用歷史訓練並統一重評，O2O 與 Dual 分別從凍結初值訓練。單一種子的量測只描述本次實驗。

## 查核已保存結果

從本倉庫根目錄執行。此入口僅需 Python 標準庫，可離線使用。

~~~powershell
python -B -X utf8 review_evidence.py verify
python -B -X utf8 review_evidence.py results
~~~

第一個命令核對交付檔案、SHA-256 與多餘檔案，第二個命令讀取已保存的 CSV，核對 E20 權重索引並列出 AP（百分比）。實體 checkpoint 可透過外部目錄另行核驗。[資料及權重](DATA_AND_WEIGHTS.md)

| 閱讀目的 | 位置 |
|---|---|
| 三組終點與逐輪指標 | [完整 Val](full_val/results.csv)、[250 張比較](evidence/validation_all.csv)、[逐輪指標](evidence/epochs_all.csv) |
| 量測結果圖 | [逐輪曲線](figures/01_逐轮曲线.png)、[兩個驗證範圍](figures/02_两个验证范围.png) |
| 模型變動、精度修復及來源授權 | [UPSTREAM](UPSTREAM.md)、[兩份補丁](patches/)、[授權](LICENSE) |
| 三組配置、歷史程式與環境 | [configs](configs/)、[tools](tools/)、[記錄環境](evidence/environment.json)、[套件版本](requirements.txt) |

## 完整歷史重現

需四份授權的同級本地工作區 FEASIBILITY、PILOT、暫停 EXP20、NUMERIC。解釋器為 FEASIBILITY 內原 Windows Python 3.10.11，PyTorch 2.0.1+cu118、torchvision 0.15.2+cu118、CUDA 11.8、NumPy 1.26.4，顯示卡為 RTX 5000 Ada Generation Laptop GPU。歷史准入還核對解釋器位置、GPU、歷史清單及凍結檔案。requirements.txt 列出實際 Python 環境的套件版本，獨立的標準庫證據核驗無須安裝套件。

先把 PowerShell 變數設為四份封存目錄的共同父目錄。plan 會檢查封存清單指紋、初值、O2M 終點、官方程式、VOC tar 及必要路徑。

~~~powershell
$historyRoot = (Resolve-Path '<四份歷史工作區的共同父目錄>').Path
python -B -X utf8 prepare_replay.py plan --history-root $historyRoot
~~~

若自行重現，使用全新的同級目錄。prepare 僅組裝本地輸入。後續研究命令會使用 GPU、寫入新工作區，並依序進行准入重評、兩組訓練、Val 與量測。下列為歷史步驟；本次交付測試範圍為完整性、資料表與前置檢查。

~~~powershell
python -B -X utf8 prepare_replay.py prepare --history-root $historyRoot
if ($LASTEXITCODE -ne 0) { throw '組裝失敗' }
Set-Location (Join-Path $historyRoot 'AT3-P2-REPLAY')
$researchPython = Join-Path $historyRoot 'AT3-P2-FEASIBILITY-01/envs/v9/Scripts/python.exe'
$env:AT3_PRISTINE = '0'
$env:YOLO_AUTOINSTALL = 'false'
$env:GIT_OPTIONAL_LOCKS = '0'
function Invoke-ResearchStep { & $researchPython -B -X utf8 -u @args; if ($LASTEXITCODE -ne 0) { throw "研究步驟失敗 $args" } }
Invoke-ResearchStep tools/resume_check.py
Invoke-ResearchStep tools/train.py O2O
Invoke-ResearchStep tools/train.py Dual
Invoke-ResearchStep tools/full.py
Invoke-ResearchStep tools/final_measure.py
Invoke-ResearchStep tools/final_analysis.py
Invoke-ResearchStep tools/final_reports.py
Invoke-ResearchStep tools/final_figures.py
~~~

同級原件由組裝器以唯讀方式使用。Excel 生成另依賴原研究的 Node 文件執行環境。評估使用 NMS／Top-K 與 eval-only PSA FP32 修復，採用單次 E20 完整 Val。跨機訓練與延遲表現尚無獨立重測。
