# 資料、權重與指紋

| 外部輸入 | 取得與放置 |
|---|---|
| VOC2007 Train/Val | 從經授權的 PASCAL VOC2007 來源取得 `VOCtrainval_06-Nov-2007.tar`。歷史資料位置為暫停 EXP20 的 `datasets/VOC2007/`，配對名單沿用該封存工程。SHA-256 `7d8cd951101b0957ddfd7a530bdc8a94f06121cfc1e511bb5937e973020c7508`，MD5 `c52e279531787c972589f7e41ab4ae64`。[VOC2007 官方來源](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/) |
| YOLOv10-N COCO 原權重 | 由官方 v1.1 release 取得 `yolov10n.pt`。歷史檔名 `weights/yolov10n-coco.pt`，SHA-256 `61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e`。[作者 release](https://github.com/THU-MIG/yolov10/releases/tag/v1.1) |
| 兩份不同初值 | 透過授權的課程研究歸檔取得 `init_single_shared.pt` 和 `init_Dual.pt`。共用檔可按同一 SHA 還原為 `weights/init_O2M.pt` 及 `weights/init_O2O.pt`。歷史組裝從 PILOT 讀取三個原名。 |
| 三份 E20 last 與三份子集 best | 向課程工程持有人取得獲准查閱的歷史檔。索引逐檔列出檔名、封存工程內路徑、bytes 及 SHA。取得 last 後可用下方命令核對實體。 |

[八個權重的逐檔指紋](evidence/weight-fingerprints.csv) · [重現所需歷史輸入](evidence/replay-inputs.json)

PowerShell 可逐檔執行 `Get-FileHash -Algorithm SHA256 -LiteralPath <檔案路徑>`。在外部目錄按索引 filename 欄命名三份 E20 檔後執行：

~~~powershell
python -B -X utf8 review_evidence.py results --weights-dir <外部權重目錄>
~~~

歷史抽樣為 Train 1,000 張、逐輪 Val 250 張、完整官方 Val 2,510 張。250 張在同一官方 Val 中。完整研究重現另需 FEASIBILITY、PILOT、暫停 EXP20 與 NUMERIC 的原始封存材料；初值與實驗終點沒有經核實的公開下載位址。本倉庫提供記錄與數值閱讀所需的全部檔案。
