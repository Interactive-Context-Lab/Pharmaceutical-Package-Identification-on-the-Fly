tf.__version__ == 1.5.0以上
scipy(version==1.0.0)
使用pycharm
ubuntu 16.04和 window10都可執行，主要重點在tensorflow環境的安裝成功與否

HBIN_part_level操作流程:
測試:
1.權重放置:
A.首先請將訓練好的BCN與RIN的權重分別放置在checkpoint資料夾裡的BCN與RIN的資料夾中
B.接著請更新HBIN_testing.py中，程式56行更新新的RIN權重檔名
C.之後請至model.py中，程式453行更新新的BCN權重檔名
(*注意如要測試HBIN_baseline對應的BCN權重為./checpoint/BCN/pix2pix.model-62)

2.影像放置:
A.請將準備好的原始藥排影像以及藥排背景相減影像放置在HBIN_bg_sub_testdata資料夾中

3.測試HBIN:
A.執行HBIN_testing.py進行測試，辨識錯誤的影像會儲存至HBIN_error資料夾中

demo:
1.權重放置:
A.首先請將訓練好的BCN與RIN的權重分別放置在checkpoint資料夾裡的BCN與RIN的資料夾中
B.接著請更新main.py中，程式56行更新新的RIN權重檔名
C.之後請至model.py中，程式453行更新新的BCN權重檔名

2.相機編號設定:
A.請正確更新main.py中，程式54、55行相機的編號

3.HBIN demo:
A.執行main.py