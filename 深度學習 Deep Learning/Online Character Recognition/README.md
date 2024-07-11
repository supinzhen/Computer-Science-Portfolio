## 題目與簡介

###  [Kaggle : Online Character Recognition](https://www.kaggle.com/competitions/online-character-recognition/overview)

此題目包含數字手寫軌跡的資料集，需要設計一個循環神經網絡來將手寫軌跡識別為相應的數字。而Kaggle將根據分類準確性來評估網路的性能。

我這次將使用Python實作此項目，並搭配Pytorch實作模型。

## 訓練資料概述
在開始訓練模型之前，對資料進行分析有助於往後對模型的設計。

由下圖可知，此訓練資料"train_in.csv"有7494筆，有17個欄位，並且皆為整數，，第一個欄位為序列碼，在資料預處理時可以把它去除，使用剩下16個欄位去做神經網路的訓練。可以看到這個資料集相當完整，並沒有缺失資料。

"train_out.csv"的部分共有兩欄，序列碼的部分缺失，在資料預處理時可以將其去除。至於Label的部分也是整數，並且沒有缺失。資料集描述可知，Label欄位的最小值為0，最大值為9，皆為整數，因此共有10種輸出。

![data_info](https://hackmd.io/_uploads/rJ-D8d_U3.png)


## 資料預處理
將"train_in.csv"與"train_out.csv"中的"Serial No."欄位去掉，前者留下x1~y8，共16個欄位，後者剩下Label。

下圖為資料預處理之後的資料集描述:

![data_info_preprocess](https://hackmd.io/_uploads/SJARf__L3.png)


## 模型設計

這次將使用循環神經網路去解決便是手寫數字軌跡的問題。

### 循環神經網路 Recurrent Neural Network (RNN) 簡介
循環神經網路（Recurrent Neural Network，RNN）是一種類神經網路模型，主要用於處理序列資料，擁有記憶和上下文依賴的能力。RNN在處理有序列性的資料時能夠捕捉到時間上的相關性，使得模型能夠更好地理解序列中的時間依賴關係。

RNN使用隱藏狀態，能在每個時間點被更新和傳遞，並擁有記得先前輸入的能力。在處理序列資料時，RNN通過將現在這個時間底點的輸入和上一時間點的隱藏狀態作為輸入，並計算新的隱藏狀態。這種遞歸的過程使得RNN能夠捕捉到時間上的動態變化。單純的RNN因為權重指數級爆炸或梯度消失問題，難以捕捉長期時間關聯，為了解決這個問題，出現了**長短期記憶(LSTM)** 網路。

> Read more about Recurrent Neural Network [here](https://zh.wikipedia.org/zh-tw/%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)

### RNN設計

在 RNN 的訓練過程中，隱藏狀態會被模型自動更新和調整，並用於下一個計算，這樣可以捕捉序列中的時間相依性和上下文資訊(如下圖一)。我這次使用了**兩層RNN**去捕捉時間動態的變化，並搭配Linear去做分類，再使用Softmax將輸出的機率調整至介於0~1之間(如下圖二)。我這次設定的hidden_size為514，選擇Tanh作為激活函數，並使用dropout機制避免模型的過度擬合。

![RNN](https://hackmd.io/_uploads/HkFjT6qDn.png)


由於是分類問題，因此損失函數這次選擇的是**交叉熵(Cross Entropy)**，優化器則是使用**平均隨機梯度下降法(Averaged Stochastic Gradient Descent)**。



## 模型訓練與訓練結果

我這次將Batch Size設置為2，，以0.0001的學習率訓練了1000 epoch。

將這次訓練出來的結果的loss繪製出來如下:

![loss_plot](https://hackmd.io/_uploads/rJegUa9wn.png)



## 問題與討論

這次遇到的問題在於在定義init_hidden函式時，常遇到大小不對的錯誤，以及不知道如何優化自己設計的神經網路(Loss 一直無法再往下降)。前者在網路上蒐集了很多的資料後才解決，後者則是經過不斷的trial and error，經過不斷調整hidden layer的數量、調整dropout的百分比，增加訓練的epoch、將非線性改為relu...等，也嘗試使用LSTM，但最終的結果並未比較好(或許也是因為我不知道怎麼優化)，因此最後還是使用傳統的RNN，經過不斷的調整之後做出了一個還行的結果。

在這個作業中學到 RNN 與 LSTM 的實際操作，從實做中也更了解到兩者是怎麼運作的。除了深度學習之外，學到最多的部分是使用python做資料處理，從前面製作Dataset，到繳交時為了符合Kaggle的格式，都在資料處理上花了一點功夫，而這個就算之後不做深度學習也可以變成一個很好用的工具。
