## 題目與簡介

使用一系列形態過濾器(morphological filters)來檢測缺失的齒輪。

![tooth](https://hackmd.io/_uploads/HyF6P3Rdn.png)

我將使用Python 實作此項目。

## 使用方法

步驟如下:
1. 輸入圖像
2. 設計一組Kernel：<b>gear_body</b>、<b>sampling_ring_spacer</b>、<b>sampling_ring_width</b>、<b>tip_spacing</b>、<b> defect_cue</b>
3. (e) 使用結構元素 <b>gear_body</b> 擴張(Opening)它以除去牙齒，使用結構元素 <b>sampling_ring_spacer</b> 擴張(Opening)它以將其帶到齒的底部牙齒，使用結構元素 <b>sampling_ring_width</b> 對其進行擴張(Opening)，以將下一個圖像帶到齒尖，並減去最後兩個結果以獲得剛好適合牙齒的環。
4. (f) 將其與原始圖像相與以生成僅包含牙齒的圖像。
5. (g) 使用結構元素 <b>tip_spacing</b> 擴張(Opening)牙齒圖像會產生實心環圖像，只要牙齒中有缺陷，該圖像就會在實心環中留有空間。
6. (h)從採樣環中減去它只留下缺陷，這些缺陷由結構元素<b>defect_cue</b>擴大。
7. 在輸出圖像上使用<b>紅色矩形</b>標記有缺陷的齒輪。

![Procedure](https://hackmd.io/_uploads/rJigs30u2.png)

### 實驗結果

![Result](https://hackmd.io/_uploads/Syj1j30dn.jpg)


## 心得與討論

這次使用侵蝕與膨脹去偵測齒輪的缺角，覺得很有趣的事情是用這兩種方法就可以做到這件事。
一開始製作出來的偵測結果有點奇怪，在不斷的嘗試調整kernel與threshold的大小與參數之後才得到比較好的效果。原本沒有想去理會圖片中的矩形框線(齒輪外圍黑與白的矩形邊界)，但後來覺得他有點礙眼所以才把它去除掉，這樣在後面偵測瑕疵輪廓位置時也容易的許多。
