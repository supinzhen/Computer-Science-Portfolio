## 題目與簡介

使用網路攝影機輸入影像，並執行自適應背景扣除(Adaptive Background Subtraction)與前景檢測(如下圖所示，圖片擷取自網路)。

![Adaptive Background Subtraction](https://hackmd.io/_uploads/SkVVehR_n.png)

## 使用方法

我這次嘗試使用了兩種方法做背景分割:
1. 使用addWeighted做動態背景，並與當前影像相減，得到在移動中的物件。
2. 使用BackgroundSubtractorKNN直接做背景分割。

### 使用addWeighted做動態背景

使用addWeight對背景做加權，讓他可以因應逐漸改變的背景。嘗試過不同的Alpha值，並發現結果設為0.95的效果最佳，據觀察，如果Alpha值設太小，背景改變得太快，比較難看出與當前影格的影像差值，而如果Alpha值設得太大，移動的軌跡會成為背景的一部分，會干擾去背的效果。

步驟如下:

1. 執行<b>自適應背景扣除</b>。 即，<i>S(t)=abs(I(t)-B(t))</i>，其中<i>B(t)</i>是自適應背景圖像。
2. 執行<b>前景檢測</b>。 即 <i>F(t)=I(t) 如果 S(t) > 閾值</i>及為前景。
3. 顯示拍攝圖像<i>I(t)</i>、自適應背景圖像<i>B(t)</i>、減影圖像<i>S(t)</i>以及 前景圖像<i>F(t)</i>。
4. 調整alpha值並觀察結果

#### 實驗結果

![addWeighted](https://hackmd.io/_uploads/Hkb9b2R_2.png)


### 使用 BackgroundSubtractorKNN

第二種使用OpenCV中的K-Nearest (KNN)背景分割器做背景分割。兩種方法都有使用高斯模糊去柔化邊緣，並使用膨脹擴增邊緣。背景置換的方法是，利用相反的遮罩將前景與背景分別做成黑底的圖片並進行疊加，原本使用np.where函式將RGB為0的地方(也就是遮罩蓋到的地方)進行與背景的像素置換，不過後來發現，他會將R、G、B單獨為0的像素也進行置換，顏色就跑掉了，因此才改成現在這種方法。

#### 實驗結果

![BackgroundSubtractorKNN](https://hackmd.io/_uploads/B113W3Ru2.png)

## 心得與討論

兩種背景分割方法的相同之處在於，在<b>前景移動時</b>都比較容易分辨前景與背景，邊緣部分也比較明顯，而色塊(如人的皮膚、衣服、頭髮等)在分辨時會比較難分割。至於效果，第二種方式比第一種還好，不過使用第二種方式沒有動態背景影像，只有分割出來的結果。
