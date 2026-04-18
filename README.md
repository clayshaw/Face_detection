# 版本差異
### version_0.0
- 偵測 : yolov26
- 辨識 : insightface
- 圖形 : PyQt5
### version_0.1
- 偵測 : yolov26
- 辨識 : insightface
- 圖形 : tkinter
- 改善 : 整合為同一個類作呼叫
### version_0.2
- 偵測 : insightface
- 辨識 : insightface
- 圖形 : tkinter
- 改善 : 偵測改為insightface
- 缺點 : 速度非常受限，使用cpu偵測約10偵，gpu約30偵
### version_0.3
- 偵測 : insightface
- 辨識 : insightface
- 圖形 : tkinter
- 改善 : 分開相機流與偵偵測流，單獨各開thread，互不影響，偵數大幅上升

#### 測試環境
- cpu : R7-6800H
- gpu : RTX 3050Ti 4GB
- mem : 40GB
- cuda : 13.0

