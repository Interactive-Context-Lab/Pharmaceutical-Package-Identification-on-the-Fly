# BCN Network
本網站的程式經由參考並修改 https://github.com/yenchenlin/pix2pix-tensorflow 之pix2pix 的內容獲得。
## 環境
tf.__version__ == 1.5.0以上

使用pycharm

ubuntu 16.04和 window10都可執行，主要重點在tensorflow環境的安裝成功與否
## 程式碼
### 重要程式:
訓練/測試/demo指令->ins.txt
### 相機編號檢查:
webcam_check.py
### 主程式:
main.py
### 模型架構:
model.py
## 訓練流程:
首先請將所要訓練的資料依train/val分成訓練集與測試集
(圖片大小為原始藥排影像256x256，語義分割影像256x256；訓練與測試影像為將兩者進行拼接為256x512的影像進行訓練與測試)
接著輸入訓練指令至pycharm中main.py的Script parameters，執行main.py
*訓練過程中，可由sample資料夾觀測目前圖片生成情況*
## 測試流程:
首先請確認main.py中488行程式碼for i in range()中所要測試的權重檔範圍
接著輸入測試指令至pycharm中的main.py的Script parameters，執行main.py
*測試時，pix2pix生成的影像會生成在test資料夾裡*
## demo流程:
先執行webcam_check.py，找到兩個相對相機的相機編號，之後將webcam_check.py關閉，並修改main.py中demo部分(程式388、389行)的相機編號
(應避免相機編號相反，務必使程式中，RTT視窗的藥排背面在左邊，藥排正面在右邊)
接著前往checkpoint資料夾修改所對應訓練的權重資料夾(你訓練的資料集檔名_batchsize_256)中的checkpoint檔案，編輯並將model_checkpoint_path修改為所要讀的權重檔名
最後輸入demo指令至pycharm中main.py的Script parameters，執行main.py
*值得注意，若要執行BCN_bg_sub請將main.py中409-420行與434-435行的程式註解打開，若要執行BCN_baseline則關閉409-420行與434-435行的程式註解*
