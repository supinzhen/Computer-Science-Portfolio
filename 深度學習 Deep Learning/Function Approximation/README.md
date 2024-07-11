## 題目與簡介

###  [Kaggle : Function Approximation](https://www.kaggle.com/competitions/function-approximation/overview)

此題目的訓練資料受到一些雜訊的干擾。需要設計一個網絡來消除。而Kaggle將計算輸出與真實輸出之間的均方誤差(Mean Square Error, MSE)，並藉此來評估網路的性能。

我這次將使用Python實作此項目，並搭配Pytorch與d2l實作模型。

> Full Code on [GitHub](https://github.com/supinzhen/Computer-Science-Portfolio/tree/f3bf9fd0ce9b5a02c1165d97a601f9bbce0a217e/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%20Deep%20Learning/Function%20Approximation)

## 訓練資料概述
在開始訓練模型之前，對資料進行分析有助於往後對模型的設計。

由下圖可知，此訓練資料有8000筆，並且有四個欄位 **'id'**、**'x1'**、 **'x2'** 與 **'y'**，其中 **'y'** 為這次要預測的欄位，而 **'id'** 則需要在資料預處理時處理掉。d可以看到'x1'與'x2'本身就介於-1~1之間，在做資料預處理的時候並不用特別將其正規化，並且可以看見要預測的y值介於1.96與0.0262之間。

![data_info](https://i.imgur.com/dG5fr7f.png)


接下來將訓練資料畫出來觀察他的分布:

![data_plot](https://i.imgur.com/xXKu7dk.jpg)

由上圖可知，此資料集並非屬於線性分布，是一個非線性的數據，使用單純的Linear Regression無法解決此問題，因此最終決定使用**多層感知機Multilayer perceptron (MLP)** 來解決此問題。


## 資料預處理
將訓練資料集中的"id"欄位去掉，並且先找出資料集中為數值資料的欄位，檢查其中有沒有遺失的資料，並將其替補成0。由於'x1'與'x2'本身就介於-1~1之間，因此並沒有特別去做資料正規化。

下圖為資料預處理之後的資料集描述:

![data_info_preprocess](https://i.imgur.com/xuneTj4.png)


## 模型設計

### 多層感知機 Multilayer Perceptron (MLP) 簡介
多層感知機(Multilayer Perceptron, MLP)為一種監督式學習的神經網路模型，至少包含三層結構，分別為: 輸入層、隱藏層和輸出層，每一層有多個節點，並都全連接到下一層的節點中。輸入層接受輸入資料，隱藏層透過**激活函數(Activation function)** 進行一系列的線性和非線性計算，輸出層則輸出最終的結果。透過不斷調整權重(wight)和偏差(bias)，讓他可以得到最佳的結果。

> Read more about Multilayer Perceptron [here](https://zh.wikipedia.org/zh-tw/%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8)

### MLP設計

PyTorch在torch.nn中提供了許多現成的Module可以使用。用來表示神經元、層甚至模型，接收輸入，更新參數，計算輸出。

我這次使用了nn.Sequential來建構模型，並搭配LazyLinear、激勵函數ReLU與Dropout機制，最終建構出一個七層的MLP模型，損失函數選擇**均方誤差(Mean square error，MSE)**，優化器選擇**隨機梯度下降法Stochastic Gradient Decent(SGD)**。

我將介紹各層的功能與我這次設計的模型。

```
self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens_1), 
            nn.ReLU(),
            nn.LazyLinear(num_hiddens_1), 
            nn.ReLU(),
            nn.Dropout(dropout_1), 
            nn.LazyLinear(num_hiddens_2), 
            nn.ReLU(),
            nn.LazyLinear(num_hiddens_2), 
            nn.ReLU(),
            nn.Dropout(dropout_2), 
            nn.LazyLinear(num_hiddens_3), 
            nn.ReLU(),
            nn.Dropout(dropout_3), 
            nn.LazyLinear(num_hiddens_4), 
            nn.ReLU(),
            nn.Dropout(dropout_4), 
            nn.LazyLinear(num_hiddens_5), 
            nn.ReLU(),
            nn.LazyLinear(num_outputs))
```

### LazyLinear
Pytorch提供了現成的nn.Linear層，可以自訂輸出與輸入維度用來進行 y=Wx+b 這類的線性運算。其中nn.LazyLinear為nn.Linear的子類別，其參與數只有輸出維度，並且在第一次調用forward時對參數進行初始化。

在Pytorch中LazyLinear的實現方式如下

```
torch.nn.LazyLinear(out_features, bias=True, device=None, dtype=None)
```

> Read more about LazyLinear in Pytorch [here](https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html#torch.nn.LazyLinear)


### Dropout 機制
Dropout為一種對抗過擬合的方法，藉由在訓練時的每一次的迭代 (epoch)用一定機率**丟棄**隱藏層的神經元，讓其不會傳遞訊息。在反向傳播時，由於被丟棄的神經元的梯度是 0，讓他不會過度依賴某一些神經元，來達到對抗過擬合的效果。不過要注意的是，只有在訓練時需要 dropout，測試時不需要。
![dropout](https://i.imgur.com/89lnwZf.png)

(圖片擷取自[網路](https://datasciocean.tech/wp-content/uploads/2022/08/dropout.png))

在Pytorch中Dropout的實現方式如下:

```
torch.nn.Dropout(p=0.5, inplace=False)
```
> Read more about Dropout [here](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout)

### 激勵函數(Activation Function)
前面所述的LazyLinear訓練時會以**線性的模式**進行求解，但就如此次的問題一樣，現實中的問題常為非線性的，因此要使用激勵函數(Activation Function)來**減少線性程度**，使得神經網路的分析結果更準確。常被使用的激勵函數包含了ReLU、Sigmoid與Tanh，我此次將使用ReLU函數。

#### ReLU (Rectified Linear Unit)
ReLU為目前最常被使用的激勵函數，數學式型態可表示為:

![ReLU](https://i.imgur.com/lEAL2VH.png)

(圖片擷取自[網路](https://ithelp.ithome.com.tw/articles/10304438?sc=iThelpR))

在Pytorch中ReLU的實現方式如下:

```
torch.nn.ReLU(inplace=False)
```

> Read more about ReLU [here](https://zh.wikipedia.org/zh-tw/%E6%96%9C%E5%9D%A1%E5%87%BD%E6%95%B0)

### 損失函數 Loss Function
真實數據與模型預測出來的數據相比相差多少，稱之為**誤差**，在訓練中，會希望誤差越小越好(最小化)，這樣的模型也會越棒。我這次將使用均方誤差 Mean Square Error (MSE)。

#### 均方誤差 Mean Square Error (MSE)
**均方誤差 Mean Square Error (MSE)** 就是一種誤差的計算方式，為預測值與真實值之間差異的均方值。算式如下:

![MSE](https://i.imgur.com/nBhliSQ.png)

(圖片擷取自[網路](https://ithelp.ithome.com.tw/articles/10216054))
    
實現的步驟為: 
1.    拿預測的值(predictd value)與真實數據(labeled value)相減
2.    所有的相減值皆平方(避免誤差正負相消)，並取總和
3.    除以總數量平均

> Read more about Mean Square Error [here](https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E8%AF%AF%E5%B7%AE) 

在Pytorch中有現成的模組可以調用:


`torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`


> Read more about Mean Square Error in Pytorch [here](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)

### 優化器 Optimizer
神經網路由許許多多的神經元所組成，每一個神經元都有自己的權重，而Optimizer的作用就是用來**幫助神經網路調整參數**。在損失函數判斷完誤差值後，Optimizer則會開始調整參數使**誤差越小越好**。我這次使用的是**隨機梯度下降法Stochastic Gradient Decent(SGD)**。

#### 隨機梯度下降法 Stochastic Gradient Decent(SGD)
梯度下降法(gradient descent，GD)是一種不斷去更新參數找解的方法，而隨機梯度下降法 Stochastic Gradient Decent(SGD)則為一次跑一個**隨機抽取的樣本或是小批次(mini-batch)**，並算出一次梯度或是小批次梯度的平均後就更新一次。

在Pytorch中有現成的模組可以調用:

```
torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False)
```

> Read more about Stochastic Gradient Decent(SGD) in Pytorch [here](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html?highlight=sgd#torch.optim.SGD)

### 交叉驗證 K-Fold 
為了避免用少料資料訓練模型所造成的過度擬合，可以使用交叉驗證在訓練資料中再分出一部分做為驗證資料，用來評估模型的訓練成果。實現方法為將原始的訓練資料分成K組(K-Fold)，並使用K-1組子集資料作為訓練集，最後一份為測試集，最終會得到K個模型。

![K-Fold](https://i.imgur.com/OH5zqge.png)

(圖片擷取自[網路](https://www.google.com/url?sa=i&url=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F67986077&psig=AOvVaw2-WBtE0RRhdD0ElMgh2Bvb&ust=1683203082771000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCMiAjqCS2f4CFQAAAAAdAAAAABAR))


## 模型訓練與訓練結果

我這次將Batch Size設置為256，以0.3的學習率訓練了300 epoch。

將測試資料丟進訓練出的模型結果如下:

![submission_plot](https://i.imgur.com/87QA3Es.jpg)


## 問題與討論

這次遇到第一個問題是在於GPU的使用，在環境部分設置了很久，不過上網查了許多資料後還是順利完成。另一個問題是在思考每一層節點數的設置，到底節點數要由多到少，還是每層一樣，我做了許多嘗試，也參考了網路上別人的做法，最後決定使用節點數由大到小的模型。這次嘗試使用K-Fold將訓練資料分批訓練及測試，也使用dropout機制避免過度擬合。不過在訓練模型時，有一段時間loss停在0.05，降不下去，嘗試過調整模型的參數、調整模型的結構，結果都差不了太多。最後嘗試將訓練好的模型參數存起來，並重複不斷讀入、訓練、存檔，才將讓loss可以繼續往下降。
