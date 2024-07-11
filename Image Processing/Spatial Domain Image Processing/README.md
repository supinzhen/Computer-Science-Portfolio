## 題目與簡介

### 第一部分: Histogram Equalization

取得過暗、過亮、低對比、高對比四張照片，找到轉換曲線(Transfer Curve)來增強這些圖像。可使用對比拉伸方法，例如功率曲線(Power Curves)、以及直方圖均衡化(Histogram Equalization)。

![four_image](https://hackmd.io/_uploads/HkHIGqR_n.png)

### 第二部分: Histogram Specification

找一張月球的圖片，圖片中大部分地區幾乎是黑暗的。使用直方圖均衡化(Histogram Equalization)與直方圖均衡化(Histogram Specification)增強圖片，並進行比較與討論。

![moon_image](https://hackmd.io/_uploads/HJFZQq0Oh.png)


我這次將使用Python實作此項目。

> Full code on [GitHub](https://github.com/supinzhen/Computer-Science-Portfolio/tree/686f6b883738500d86ef55201f4678e64a81b34e/%E5%BD%B1%E5%83%8F%E8%99%95%E7%90%86%20Image%20Processing/Spatial%20Domain%20Image%20Processing)

## 目錄

[TOC]

## 第一部分: Histogram Equalization

我這次使用的是我自己拍攝的石頭照片，使用修圖軟體Adobe Photoshop調整曝光及對比得到四張影像：(a)Dark、(b)Bright、(c )Low Contrast、(d)High Contrast。

影像原圖與直方圖分別如下：

![image_his](https://hackmd.io/_uploads/SkQ8Xc0d3.png)

這次嘗試了許多方法去調整上列圖片，有直接套用所有圖片的轉換方式也有針對不同圖片所採取的不同轉換方式，將列於下列表格之中：

| 影像 | 調整方式 |
| -------- | -------- |
| (a)Dark |Histogram Equalization、Log Transformation、Power Law Transformation|
|(b)Bright|Histogram Equalization、Inverse Log Transformation、Power Law Transformation|
|(c )Low Contrast|Histogram Equalization、Contrast Stretching|
|(d)High Contrast|Histogram Equalization、Contrast Stretching|


我將先呈現Histogram Equalization對每一張圖片轉換呈現出來的結果，再針對每一張照片所使用的方法做陳述，並在最後進行對所有方法的比較。

### Histogram Equalization直方圖均衡化 - 針對所有圖片所進行的調整


Histogram Equalization直方圖均衡化，為運用累積分布函數對原始影像的灰值進行調整，將比較集中的灰值區間變成全部灰度區間的均勻分布。
可以將過暗或背光的影像進行對比度的調整。OpenCV有cv2.equalizeHist 可以調用，不過這次不會使用此函數，而是自己寫。

轉換過程程式碼如下：

```
# 取得此張圖片的原始直方圖
    his_cur, bins = np.histogram(img.flatten(),256,[0,256])
    
    # 得到每個灰階的累積和
    his_normalized = his_cur.cumsum()
    
    # Equalization
    his_normalized = (his_normalized - his_normalized.min()) / (his_normalized.max() - his_normalized.min()) * 255
```

#### 實驗結果(a)Dark

對於每一張圖片會顯示原始圖片與原始圖片的直方圖，以及直方圖均衡化後的圖片、直方圖，以及他的轉換曲線，並進行觀察。

首先是將(a)Dark進行直方圖均衡化(如下圖)，由於原本的灰度像素值分布於直方圖的左側(灰度值都在150以下)，在直方圖均衡化之後會將他的對比度拉開，並將灰度值整體往直方圖右邊移。轉換曲線的部分，可以看到因為原本圖片並沒有灰度150以上的部分，所以所有的匹配是將原本圖片150度以下的值匹配到新圖的255以下。整體轉換結果為將亮度變亮並增加對比度。

![dark_his](https://hackmd.io/_uploads/B1l-BcA_h.png)

#### 實驗結果(b)Bright

再來是將(b)Bright進行直方圖均衡化(如下圖)，由下圖可知，原本的灰度像素值分布於直方圖的右側，屬於偏亮的圖片，在直方圖均衡化之後會將他的對比度拉開，並將灰度值整體往直方圖左邊移，將圖片壓暗。轉換曲線的部分，可以看到直方圖均衡化將圖片的像素壓暗。整體轉換結果為將亮度變暗並增加對比度。

![bright_his](https://hackmd.io/_uploads/HJj8S5CO3.png)

#### 實驗結果(c)Low Contrast

第三張圖片是將(c)Low Contrast進行直方圖均衡化(如下圖)，原本的灰度像素值分布於直方圖的中間，屬於對比度較低的圖片，在直方圖均衡化之後會將他的對比度拉開，增強對比。轉換曲線的部分，可以看到直方圖均衡化將圖片的像素亮部增強、暗部更暗。整體轉換結果為將圖片增加對比度。

![low_his](https://hackmd.io/_uploads/rkLAHcAuh.png)

#### 實驗結果(d)High Contrast
最後，將(d)High Contrast進行直方圖均衡化(如下圖)。從轉換曲線來看，直方圖均衡化對此圖片的影響較少。

![high](https://hackmd.io/_uploads/HkGz89AOn.png)

將上述四張圖的轉換結果放到一起來看，直方圖均衡化將四張亮度、對比不同的圖片，分別以不同的轉換曲線對圖片做直方圖均衡化，讓他看起來
有相似的結果。

![his_compare](https://hackmd.io/_uploads/BkG6U5Adn.png)

### Log Transformation 對數變換與 Inverse Log Transformation 逆對數變換 - 針對 (a)Dark 與 (b)Bright 所做的調整

#### Log Transformation對數變換–針對(a)Dark調整

對數變換主要對影像低部灰值進行擴展，並將影像高部灰值進行壓縮，適合用於過暗的影像。變換公式如下：

![log_math](https://hackmd.io/_uploads/ByDUDqCu3.png)

#### 實驗結果
轉換結果如下圖，可以得知，雖然對數變換可以讓整體亮度變亮，但對影像對比度的調整有限，轉換出來的圖片對比對偏低。

![dark_log](https://hackmd.io/_uploads/SyDuP5ROn.png)

#### Inverse Log Transformation逆對數變換 – 針對(b)Bright調整

逆對數變換主要對影像高部灰值進行擴展，並將影像低部灰值進行壓縮，為對數變換的反函數。適合用於調整過亮的影像。

#### 實驗結果
轉換結果如下圖：

![bright_invLog](https://hackmd.io/_uploads/ByWCv90_2.png)

### Power Law Transformation – 針對(a)Dark與(b)Bright所做的調整

透過公式，可以調整照片的明亮度，讓過曝的照片暗一點，讓曝光度不足的照片亮一點，並且可以透過調整r，改變調整的程度。

變換公式如下：
![powerLog_math](https://hackmd.io/_uploads/rJiHdcC_h.png)

* 當r < 1時，將低亮度的灰階值拉高，使整體照片變亮，適合用在過暗的圖片。
* 當r = 1時，整體圖片將不會進行任何改變。
* 當r > 1時，將高亮度的灰階值壓暗，使整體照片變暗，適合用在過亮的圖片。

我將使用(a)Dark與(b)Bright兩張圖，並透過調整不同的r值，觀察Power Law Transformation的效果。

#### 實驗結果 (a)Dark

![dark_powerLaw_1](https://hackmd.io/_uploads/SkD_OqCd2.png)

![dark_powerLaw_2](https://hackmd.io/_uploads/HJa__qCd3.png)

由上圖可知，Power Law Transformation雖然可以提高亮度，但是對於圖片的對比度並不會進行任何調整。以(a)Dark這張圖來看，果然r值越小，調整的幅度就越高，調整完的畫面也就越亮，不過調整完的圖片對比度很低，而Power Law並不能對這部分再進行調整，不過撇開對比，我認為r = 0.3對亮度有最好的調整效果。

#### 實驗結果 (b)Bright

下圖為對(b)Bright做Power Law Transformation的結果，由實驗結果可知，r值的設置大於1，才能將過量的圖片壓暗，而r值越大，所壓的幅度也越大。嘗試的結果，我認為r = 2.5有最好的轉換效果。

![bright_powerLaw_1](https://hackmd.io/_uploads/Bkp6dqA_3.png)
![bright_powerLaw_2](https://hackmd.io/_uploads/rkf0dcC_2.png)

### Contrast Stretching 對比度拉伸 – 針對(c)Low Contrast與(d)High Contrast所做的調整

Contrast Stretching對比度拉伸為使用分段線性函數對像素的亮度進行映射匹配。
(如下圖，圖片擷取自網路)

![Contrast Stretching](https://hackmd.io/_uploads/B1f19q0_2.png)

分段線性函數程式碼如下:
```
y1 = s1 / r1 * x1
y2 = (s2 – s1) / (r2 – r1) * x2 + s2 - (r2 * (s2 – s1) / (r2 – r1))
y3 = (255 – s2])/(255 - r2) * x3 + 255 * (1 - ((255 – s2)/(255 - r2)))
```

#### 實驗結果 (c)Low Contrast

首先對Low Contrast進行實驗(共試了兩組不同的拉伸點)，期望將此圖片的對比度拉低，實驗結果:
第一組拉伸點：x1 = [79, 61]，y2 = [155, 195]
第二組拉伸點：x1 = [75, 25]，y2 = [175, 225]
由實驗結果可知，第二組的拉伸結果會使對比度更高，因為將暗部拉得更暗，並把亮度變得更亮。

![low_CS](https://hackmd.io/_uploads/SkRNq5RO2.png)

#### 實驗結果 (d)High Contrast

再來對High Contrast進行實驗(共試了兩組不同的拉伸點)，期望將此圖片的對比度拉低，實驗結果:
第一組拉伸點：x1 = [62, 106]，y2 = [191, 177]
第二組拉伸點：x1 = [50, 100]，y2 = [200, 150]
由實驗結果可知，第二組的拉伸結果會將對比度條的更低，因為將亮部壓得更低，並把暗部變得更亮。

![high_CS](https://hackmd.io/_uploads/HyRP550_2.png)

### 第一部分小結

Log Transformation對數變換與Power Law Transformation都是針對亮度去做整體的調整，對於對比度並不能做額外的控制，而Contrast Stretching 則可對對比度做拉伸調整，因此嘗試了將(a)Dark圖片做Power Law Transformation與Contrast Stretching搭配使用，結果比只用一種方法好。總而言之，在嘗試了不同的轉換方式之後，我認為單一方法或許不能讓圖片有最好的結果，這時可以嘗試用不同方法搭配使用，當然，甚麼是”最好的結果”也是非常主觀的。

## 第二部分: Histogram Specification

我在網路上找了一張月球的照片，影像原圖與直方圖分別如下：

![moon_ori+his](https://hackmd.io/_uploads/HyDwsc0_3.png)

這次嘗試了許多方法去調整上列圖片，希望能讓人更容易觀察月球表面的細節，將列於下列表格之中：

| 影像 | 調整方式 |
| -------- | -------- |
|Moon|Histogram Equalization、Histogram Specification、Laplacian Filter、Image Negatives|

我將說明各項轉換方法與其結果，並在最後進行比較。
由於Histogram Equalization直方圖均衡化已經在第一部分提過，我們這裡先來看將Moon直方圖均衡化的結果：

![moon_his](https://hackmd.io/_uploads/ryQio5Rdn.png)

### Histogram Specification直方圖匹配

Histogram Specification直方圖匹配，為將圖片進行轉換，並對特定的直方圖做匹配。可以說成將圖片的直方圖分布狀況，轉換成指定圖片的直方圖像素分布，讓原本的圖片有指定圖片的像素分布，也就可以將原圖的明亮程度、對比度等，調整成與特定圖片一致。

具體方法為，將原圖與指定圖片分別進行直方圖均衡化，再將均衡化後的原圖對應到均衡化後的指定圖片的像素值，再去尋找此像素值在為原本的指定圖片中所對應到的像素值，將他當成直方圖匹配的結果，對原圖的每一像素進行匹配，得到的就會是直方圖匹配後的結果圖。(如下圖所示，圖片擷取自網路)

![spec](https://hackmd.io/_uploads/HygRj5ROn.png)

#### 實驗結果

第一個嘗試的指定圖片為(a)Dark，由下圖可知，雖然兩張圖的總像素量不同(因為圖片大小不一樣)，但是直方圖的分布情況已經被匹配成一致。

![spec_dark](https://hackmd.io/_uploads/S1Jen5CO3.png)

為了看清楚月球表面的細節，我再將Moon與(b)Bright與(d)High Contrast進行直方圖匹配：

![spec_high_bright](https://hackmd.io/_uploads/ByeFXn5AO2.png)

將結果圖放大來看:

![spec_high_bright_ori](https://hackmd.io/_uploads/rySPn9C_h.png)
左圖:與(b)Bright進行直方圖匹配的結果。
右圖:與(d)High Contrast進行直方圖匹配的結果。

由觀察可知，月球表面的凹凸在與高對比度圖片進行匹配時突顯了出來，紋路也更加明顯。

### Laplacian Filter拉普拉斯算子

Laplacian Filter拉普拉斯算子以遮罩(Mask)的方式對影像做捲積，可以將影像的邊緣輪廓強化。其概念是梯度的散度,針對X軸與Y軸分別進行二次偏微分，然後相加。我這次使用3*3的遮罩，透過空間濾波(Spatial Convolution)對整張影像進行處理。

```
Kernal = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
```
#### 實驗結果

![lap](https://hackmd.io/_uploads/SkgyTcAdh.png)
左圖:Laplacian的結果。
右圖:將Laplacian的結果與原圖進行疊加，得到邊緣銳化的結果。

據觀察，右圖的邊緣明顯比原圖銳利許多，更能看清楚月球表面的紋路。為了使細節更容易觀看，我將邊緣銳化的結果進行了負相處理。

### Negative Image 負相影像

Negative Image 負相影像，具體方法就是將原圖減去255，將黑白倒換，得到負相影像。雖然並沒有做甚麼太複雜的處理，但在特定情況下比較易於人眼觀察。

#### 實驗結果

![nev](https://hackmd.io/_uploads/BytX65Au3.png)

據觀察，月球左上的紋理部分，負相影像比原圖更容易看到表面的紋理，而在月球與陰影的交界，負相影像的孔洞也更加的明顯。至於右上有一個類似微笑的弧形，在負相影像看來也明顯了許多。

### 第二部分小結

直方圖匹配，可以將圖片擁有別張圖片的灰度像素分布，有點像風格的轉換，不過，要找到合適的直方圖讓原圖的細節得以呈現，需要不斷的嘗試不同的灰度分布。而Laplacian的影像銳化，我認為對於想要觀察細節，或是微失焦的圖片增強很有幫助。負相影像雖然轉換簡單，但由於人眼感官的緣故，對於細節的分辨與觀察也有莫大的幫助。

## 心得與總結

在這次作業之前就有使用過修圖軟體去做影像的亮度、對比度、銳利化等調整，但都是使用別人做好的功能，並不了解其中的原理。但這次的作業使我更了解影像處理的各種基礎原理，也覺得非常好玩有趣。

在不斷的嘗試之下，我認為沒有一種影像處理方式可以解決所有的問題，不同的方式搭配、補足，才可以讓圖片符合需求。
