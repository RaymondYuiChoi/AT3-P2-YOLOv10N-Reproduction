# YOLOv10 原始碼與研究差異

本研究使用 THU-MIG 的 YOLOv10 v1.1，固定提交為 `799ff3be47d21173bcf29b351820d4b8e955e0fe`。[原始碼樹](https://github.com/THU-MIG/yolov10/tree/799ff3be47d21173bcf29b351820d4b8e955e0fe) · [授權原文](https://github.com/THU-MIG/yolov10/blob/799ff3be47d21173bcf29b351820d4b8e955e0fe/LICENSE)

| 位置 | 作用 |
|---|---|
| `research/yolov10/at3_variant.py` | 單頭消融。O2M 使用 TAL Top-10 和類別感知 NMS，O2O 使用 Top-1，梯度可回到共享特徵。 |
| `research/yolov10/ultralytics/nn/modules/block.py` | 評估時將 PSA 注意力內部運算切到 FP32，再轉回原 dtype。訓練沿用原精度路徑。 |
| `patches/` | 同一固定提交的兩份來源補丁。以授權程式快照或補丁擇一還原這兩項修改。 |
| `tools/`、`configs/` | 歷史實驗的程式與凍結配置，涵蓋配對訓練、驗證、分配觀測、成本與結果審計。 |

官方 Dual 的分支、detach 及推理 Top-K 均來自上述固定提交。倉庫保存兩個研究變動檔的完整內容及上游 LICENSE。供閱讀的 block.py 整理了六處上游既有的行末空白，並採用 LF。歷史組裝從潔淨本地工作樹的固定提交套用兩份原始補丁，還原原始換行與空白，再逐檔對照准入記錄的 SHA-256 及 NUMERIC 研究樹的 Git 差異。比較同一提交時，預期只有 block.py 受修改和 at3_variant.py 新增。

[授權文件](LICENSE) · [來源與分發說明](NOTICE.md)
