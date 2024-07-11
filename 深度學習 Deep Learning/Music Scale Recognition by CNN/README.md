## 題目與簡介

###  [Kaggle : Music Scale Recognition by CNN](https://www.kaggle.com/competitions/music-note-recognition-by-cnn)

此題目的需要識別音樂聲音的音階。將使用已被轉換成梅爾倒頻譜的二維圖像構建一個CNN模型，用來識別圖像的label，也就是圖像的音階。

![梅爾倒頻譜](https://hackmd.io/_uploads/rJtN6Lgrh.png)

我這次將使用Pytorch實作此項目。

## 訓練資料概述
此次的訓練資料共有2846張圖片，與一個truth.txt的各圖片Label檔案。圖片分別為520 x 394 pixel的png檔，而truth.txt中共有兩個欄位，第一個欄位為圖片檔名，第二個欄位為音階的label，分別在0~87之間。

比較特別的是，訓練圖片為png檔，共有4個Channel，由於第四個Channel為alpha值，此次是由梅爾倒頻譜的圖樣，也就是顏色分布，去判斷音階，並不會用到alpha channel，因此我這次將alpha通道去除，並使用3 channel的圖片去做訓練。

我將用此2846張圖片與相對應的Label訓練一個CNN模型，並匯入測試資料預測音階Label。

![data-describe](https://hackmd.io/_uploads/SyYlGPer3.png)

## 卷積神經網絡 Convolutional Neural Network (CNN)簡介
卷積神經網絡（CNN）是一種常用於影像處理和電腦視覺的神經網絡模型。它主要由捲積層、池化層和全連接層等結構所組成，能夠自動學習和提取圖像的特徵，適應各種圖像的變異性和複雜性。CNN在電腦視覺領域取得了重大突破，例如圖像分類、目標檢測和圖像生成等。

CNN使用捲積層通過應用各種不同的卷積核來遍歷圖像，進而抓住圖像的各種特徵。這些卷積核能夠捕捉圖像中的細節，同時也能夠**保持圖像的空間結構特徵**。並使用池化層縮減特徵圖的維度，以保留重要的特徵。最後將這些特徵轉換成一維向量，並通過全連接層來進行分類、預測。

> Read more about Convolutional Neural Network [here](https://zh.wikipedia.org/zh-tw/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)

## CNN設計

我這次使用了**兩次卷積層+池化層**去做特徵提取與特徵整理，並將資料攤平，進入三層的神經網路做最後分類。在過程中會使用ReLU函數增加類神經網路訓練時的非線性程度，以達到更好的分類效果。損失函數使用的是**交叉熵(Cross Entropy)**，優化器則是使用**隨機梯度下降法 Stochastic Gradient Decent(SGD)**。

我這次設計的CNN結構圖示如下:

![my-cnn](https://hackmd.io/_uploads/BkHA2teBn.jpg)

我將一一介紹各層的功能。

### Convolution Layer 卷積層

卷積層的作用是**提取輸入數據中的局部特徵**。

卷積層透過在卷積核對輸入影像的不同位置進行卷積。將卷積核滑動遍歷輸入影像，每次取得一個局部區域，並將卷積核的值與該區域進行逐元素相乘再相加，得到輸入影像的局部特徵，也就是特徵圖，例如邊緣、紋理或形狀等。並通過權重共享和平移不變性使得模型能夠更好地捕捉和識別圖像或視覺數據中的特徵。如下圖所示:

![Convolution Layer](https://hackmd.io/_uploads/SJHshjeBn.png)
(圖片擷取自[網路](https://iq.opengenus.org/conv2d-in-tf/))


在Pytorch中Conv2D的實現方式如下

```
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

> Read more about Conv2D in Pytorch [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)


### Pooling Layer 池化層

池化層的最著要作用是**減少卷積特徵的空間大小**。它分成兩種，Max Pooling 與 Average Pooling，前者為回傳區域內的**最大值**，後者為回傳區域內的**平均值**。池化層卷積層提出的重要資訊保留，並捨棄不重要的資訊，以此增加模型的收斂跟穩定性。

![Pooling](https://hackmd.io/_uploads/BkOGQngr2.png)

(圖片擷取自[網路](https://tvm.d2l.ai/chapter_common_operators/pooling.html))

在Pytorch中 MaxPooling 的實現方式如下:

```
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

> Read more about MaxPool2D in Pytorch [here](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d)

### Fully Connected Layer 全連接層

在做完特徵擷取與整理之後，將前面的結果攤平，接到基本的神經網絡中，我這次設計了三層的MLP，並搭配ReLU函數的使用，最終輸出與Label類別一樣個數的機率。

### 損失函數 Loss Function
真實數據與模型預測出來的數據相比相差多少，稱之為**誤差**，在訓練中，會希望誤差越小越好(最小化)，這樣的模型也會越棒。均方誤差 Mean Square Error (MSE)在回歸問題時非常好用，但在分類問題上交叉熵(Cross Entropy)更勝一籌，因此**我這次將使用交叉熵(Cross Entropy)**。

#### 交叉熵(Cross Entropy)

交叉熵可以衡量模型的預測概率與實際標籤之間的差距。

假設我們有一個有K個樣本的分類問題，每個樣本有C個可能的類別。對於每個樣本k，我們有實際標籤的概率分佈為y，模型的預測概率分佈為ŷ。則交叉熵的公式如下：
![cross-entropy](https://hackmd.io/_uploads/SJura2xSn.png)

(圖片擷取自[網路](https://vitalflux.com/mean-squared-error-vs-cross-entropy-loss-function/))

當預測概率與實際標籤相符時，交叉熵的值會接近於0；而當它們之間差異越大時，交叉熵的值會增加。在訓練過程中，我們希望**最小化交叉熵損失**，以使模型的預測能夠更準確地對應到實際的類別標籤。

> Read more about Cross Entropy [here](https://zh.wikipedia.org/zh-tw/%E4%BA%A4%E5%8F%89%E7%86%B5) 

在Pytorch中有現成的模組可以調用:

```
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean', label_smoothing=0.0)
```

> Read more about Cross Entropy in Pytorch [here](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)


## 模型訓練與訓練結果

這次使用batch size 64、0.005的learning rate，訓練45個epoch就有不錯的效果，loss最後降至0.0002。

## 問題與討論

這次花最多時間在於如何製作圖片資料集，以及最後在訓練TEST DATA的時候，在神經網路的輸出格式那裏花了很多時間去研究，雖然一開始卡了有點久，但在爬了一大堆文之後有都有順利完成。

比較有趣的是我畫完我設計的CNN結構圖之後，有一種"原來是這樣啊~"的感覺，以往看到好多層的神經網路結構圖都不是很明白，但經過這堂課、第一次作業到這次第二次作業，我覺得我有更了解神經網路各層的特點與作用，也更明白在PYTORCH中整個神經網路是怎麼運行的。
