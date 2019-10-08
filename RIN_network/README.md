# RIN Network
## 環境
* tf.__version__ == 1.5.0以上
* 使用pycharm
* ubuntu 16.04和 window10都可執行，主要重點在tensorflow環境的安裝成功與否
## RIN
### 訓練:
1. 資料集準備:
   A. 首先請將切分好訓練與測試的RTT影像放置在datasets/blister_pack_RTT/images/train 和 datasets/blister_pack_RTT/images/test中，
   並依照"類別_01_張數號碼"進行命名，例如:第1類的第1張藥排影像則命名為000_01_000001.jpg
   B. 執行data/prepare_tfrecords.py製作train與test的tfrecord檔

2. 訓練RIN:
   A. 請根據使用者需求調整train.py中適當的訓練參數
   B. 執行train.py


3. 測試:RIN
   A. 請調整test.py中的測試參數
   B. 請更新test.py中，程式第200行的權重讀取路徑
   C. 執行test.py
