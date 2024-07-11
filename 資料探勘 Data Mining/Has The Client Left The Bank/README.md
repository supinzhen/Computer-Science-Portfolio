## 題目與簡介

使用決策樹並選擇另一種分類方式去預測資料集中的”Churn”欄位是”0”還是”1”，也就是顧客是否離開了銀行，並使用混淆矩陣衡量分類模型的準確性。

![data](https://hackmd.io/_uploads/S1j8o2ZY2.png)

我們在決策樹的部份使用了Gini與Entropy兩種度量方式各建造了一棵樹，並選擇Random Forest做為第三種分類方式。在衡量模型方面，我們使用了10-fold cross validation 與混淆矩陣去衡量模型的效能。

我們使用R語言去做資料集介紹與建造隨機森林，決策樹則是使用Python。

![method](https://hackmd.io/_uploads/H1xG32-Y2.png)


## 資料集介紹

將資料檔案”Bank_Customer.csv”資料檔匯入名為data的資料框架中，使用函式str()讀取資料，從顯示出的資訊可知下列五種訊息：資料數為10000筆、共有12個欄位、各欄位名稱、其資料型態和儲存內容。
其中，”country”與”gender”為非數值資料，為了方便計算，在資料預處理時需要將其轉為數值資料。

![data_intro](https://hackmd.io/_uploads/r19UhnZKn.png)

接下來，資料檢查下述四種狀況：
1.	有無重複資料，以customer_id欄位來做判斷
2.	有無資料為"?"，錯誤資料內容
3.	有無資料為"NA"，無法顯示資料內容
4.	有無空白資料

![check](https://hackmd.io/_uploads/SyDuh2bth.png)

也可以使用skim()函式檢查資料(如下圖所示):

![data_skim](https://hackmd.io/_uploads/S1aFn3bYn.png)

從n_missing欄位與complete_rate欄位可知，此資料集內容相當完整。

在此資料集中，年齡平均為39歲，如圖1所示。男女比為120.12（每百女子所當男子數），如圖2所示。

![plot1](https://hackmd.io/_uploads/B1WCn2-K2.png)

我們將資料集中的六個欄位與”churn”的關係圖畫了出來，大致理解了他們與目標的關係。

![plot2](https://hackmd.io/_uploads/H1QeT2Wt2.png)

![plot3](https://hackmd.io/_uploads/ryJ7p3-t3.png)

## 描述問題

使用資料集Bank_Customer.csv，將70%的資料作為訓練資料(Training data)與30%做為測試資料(Testing data)，訓練出一個模型，預測顧客流失率(Churn，有”1”與”0”兩種結果)，並使用10-fold cross vaildation 去衡量此模型的效能。

![churn](https://hackmd.io/_uploads/HJ5wT2ZKn.png)

## 研究方法

### 決策樹Decision Tree(使用Gini度量)

決策樹是一種監督式學習方法，用於分類和回歸。透過從數據中學習推斷出的關聯規則，創建一個模型來預測目標變量的值。輸入的資料會從樹根開始走訪節點，並根據節點中的條件判斷決定此資料要走往的分支，等到走到最後一個不能再往下走的節點，該節點就是他所屬的分類。

**使用Gini Impurity(吉尼不純度)度量:**
如果數據集D包含n個類別，而 pj 是類別j在D中的頻率，則gini指數(gini(D))定義為:

![gini](https://hackmd.io/_uploads/HJhyAnWFh.png)

如果將數據集D在A上分為兩個子集D1和D2，則基尼係數 gini(D)定義為:

![gini2](https://hackmd.io/_uploads/B1VzC2WYn.png)

不純度的變化:

![gini3](https://hackmd.io/_uploads/BkCXR2bY3.png)

將所有可能的屬性枚舉出分裂點計算Gini Impurity，如果提供最小gini split(D)或不純度減少最大則會被選擇成分裂節點。

### 決策樹Decision Tree(使用Entropy度量)

**使用Entropy(熵)度量:**

尋找資訊增益最高的屬性。
對資料集D中的元組分類所需的Entropy(熵)：

![entropy](https://hackmd.io/_uploads/BJx90nZY3.png)

對資料集D進行分類所需的資訊（在使用A將D分成v個分區之後）：

![entropy2](https://hackmd.io/_uploads/r1Z202ZK2.png)

屬性A分支所獲得的資訊增益(Information Gain):

![entropy3](https://hackmd.io/_uploads/HJL0RnWK2.png)

將所有可能的屬性枚舉出分裂點計算資訊增益(Information Gain)，如果資訊增益最大則會被選擇成分裂節點。

### 隨機森林Random Forest

隨機森林是常見的機器學習演算法之一，主要是基於決策樹的組合演算法，也就是使用 CART 樹，再透過 Bagging演算法建置不同的樹，最後平均每棵樹的預測結果，如下圖。

![Random Forest](https://hackmd.io/_uploads/B1Axka-Fh.png)

（圖片來源： Chung-Yi ，ML入門（十七）隨機森林(Random Forest) ）
#### OOB（Out Of Bag）

隨機森林可以透過 OOB的樣本誤差值來做為驗證方式，其驗證方式類似於交叉驗證，是透過 Bagging 的方法，利用 bootstrap的性質來生成各個不相同的訓練集來建立分類器，原始樣本中會有部分資料被抽出不納入訓練中，而這些被抽出的資料就被稱為 Out Of Bag，再利用這些計算score，公式如下：

![OOB](https://hackmd.io/_uploads/S11SkTWK3.png)

OOB結果可估算樹的平均誤差估計，也可以用來計算單個特徵的重要性。其優點是較交叉驗證來的節省資源。

## 實驗內容

### I.	訓練決策樹Decision Tree - Gini Index

#### 步驟一、資料預處理:

將不需要用到的資料(customer_id)移除，並將兩個非數值型資料(country、gender)轉換成數值型資料，轉換方式如下:

country共有三種: France, Germany, Spain

![preprocess1](https://hackmd.io/_uploads/rJC916bt2.png)

gender共有兩種: Male, Female

![preprocess2](https://hackmd.io/_uploads/Hyi6kpZth.png)

![preprocess3](https://hackmd.io/_uploads/ByYyx6ZY2.png)

#### 步驟二、訓練資料與測試資料

將70%的資料做為訓練資料，並將30%的資料做為測試資料。
我們使用隨機的方式，將資料集中的一萬筆資料分成7000筆訓練資料，3000筆測試資料。

#### 步驟三、訓練決策樹 - Gini Index
使用scikit-learn 1.2.0中的DecisionTreeClassifier製作模型，預設度量方式為”Gini”，因此不需要額外設定。再輸入訓練資料訓練模型。

![code_gini](https://hackmd.io/_uploads/rJ7Xep-Fn.png)

訓練完後將樹畫出來看一下。

![gini_tree](https://hackmd.io/_uploads/HkDLlaZF3.png)

由於沒有設定max_leaf_node的數量，決策樹會持續分支到Gini=0才會停止，因此會訓練出一棵非常大的樹。

#### 步驟四、衡量模型 - Gini Index

交叉驗證10-fold Cross Validation: 0.79
混淆矩陣Confusion Matrix: 

![gini_confusionMatrix](https://hackmd.io/_uploads/rks9gaZt3.png)

#### 步驟五、優化決策樹

設定max_leaf_nodes，控制樹生長的大小。
使用for迴圈測試設定多少會得到最佳的分數。

使用測試出來的結果去訓練決策樹，並得到關聯規則:

![enchance](https://hackmd.io/_uploads/BkVRg6Wt2.png)

![relation](https://hackmd.io/_uploads/S1le-6-F3.png)

將決策樹視覺化:

![tree_ench1](https://hackmd.io/_uploads/Skifbp-Y3.png)

#### 步驟六、衡量優化的決策樹

交叉驗證10-fold Cross Validation: 0.859
混淆矩陣Confusion Matrix: 

![confusionMatrix_maxNode](https://hackmd.io/_uploads/By8r-aZKh.png)

### II.	訓練決策樹Decision Tree – Entropy

使用scikit-learn 1.2.0中的DecisionTreeClassifier製作模型，將度量方式設定為”Entropy”，再輸入訓練資料訓練模型。

![code](https://hackmd.io/_uploads/B1w2-pbY3.png)

其他步驟與Gini的決策樹一致，因此在這裡直接描述結果。

得到關聯規則:

![relation](https://hackmd.io/_uploads/rJ_tWa-Fh.png)

視覺化決策樹:

![tree_ench2](https://hackmd.io/_uploads/B1R5bpWt2.png)

#### 衡量決策樹 – Entropy

交叉驗證10-fold Cross Validation: 0.86
混淆矩陣Confusion Matrix: 

![confusionMatrix_entropy](https://hackmd.io/_uploads/BJfgf6-K3.png)

### III.	訓練隨機森林Random Forest

#### 步驟一、資料預處理
首先將不必要的資料刪除，也就是customer_id的部分先拿掉不納入計算。並將資料集分成70%的訓練資料和30%的測試資料。 

![](https://hackmd.io/_uploads/BJ6QMaZt2.png)

#### 步驟二、建立隨機森林
使用randomForest library建立隨機森林模型，設定taget和資料集選取，proximity設定為TRUE表示估計樣本間的相似度，mtry設定將要抽多少個變數去建立每棵決策樹，初次建立先假定抽取所有變數，importance設為TRUE ，用來觀察重要的變數（程式碼如下圖）。 

![](https://hackmd.io/_uploads/H104GT-t2.png)

預設建立樹的模型數量為500棵，其OOB（Out Of Bag）為14.97%。 

![](https://hackmd.io/_uploads/ryorG6bF3.png)

Accuracy為85.7%。 

#### 步驟三、取得最佳 mtry

使用tuneRF函式來得到最佳的mtry，預設樹的數量為500棵，計算多少mtry的OOB最低，根據下圖可知最低為3。 

![](https://hackmd.io/_uploads/SJb5fp-Fh.png)

#### 步驟四、最佳mtry套入模型

在得到最佳mtry後重複步驟二動作將mtry改為3，其他設定維持不變，可得到結果OOB從之前的14.97%下降為13.81%，結果更為準確， Accuracy為86.23%。 

![](https://hackmd.io/_uploads/ryasz6-t2.png)

#### 步驟五、算出重要特徵：
使用importance函數來得到重要特徵有哪些， MeanDecreaseAccuracy這項欄位代表，去除這項解釋變數的話，模型的準確度會減少多少%。Mean Decrease Gini代表衡量變數重要性指數，表示Gini係數減少的平均值。如下圖： 

![](https://hackmd.io/_uploads/rkbazTWth.png)

![](https://hackmd.io/_uploads/ry0RG6ZF2.png)

#### 步驟六、測試模型樹的數量

從步驟四可看出在更改mtry後，模型準確度提高，若更動樹的變數值，假定樹從預設500棵設定為 1000棵，測試運算結果是否準確度會提升。 
維持步驟四函式，僅更動樹的數量ntree=1000，其OOB為13.9%，和500棵樹的13.81%相比反而 OOB上升且消耗資源更多， Accuracy為86.2%和500棵樹的86.23%相差不遠，可知準確度並不會因為樹的量增加而提高。

## 研究結果

### I.	決策樹Decision Tree
在研究方法中一共測試了三種不同的方法，首先是使用Gini Impurity作為度量方式的決策樹。得到的準確率為79.9%。使用Gini Impurity並設置max_leaf_nodes的決策樹的準確率為86.1%。由此可知，第一種方式雖然決策樹長的比較大棵，但並沒有比較準確，反而有點過度擬合的情形，第二種方法所建出來的樹雖然較小棵，但準確率卻較高。

第三種方法為使用Entropy作為度量方式，並設置max_leaf_nodes所建出來的決策樹，準確度為85.9%，準確率與第二種方式所得出來的值差不多，但F1值第三種較高。

![](https://hackmd.io/_uploads/ryMB7TbK2.png)

### II.	隨機森林

在研究方法中一共測試了三種不同的森林，首先是樹為預設值設定 500棵，mtry將所有變數納入計算，取得結果Accuracy為85.7%。在透過函數tuneRF取得最佳mtry後，將mtry更新為3，其餘設定保持不變，取得結果Accuracy為86.23%。最後更改樹的模型數量為 1000，其餘設定不做更動，其結果Accuracy為86.2%。

由下表可知，抽取所有變數的mtry表現最差，1000棵樹雖和最佳mtry結果相近，但消耗資源較多，不符合效益，因此結果最好為最佳mtry。
從結果可證，隨機森林需配合最佳mtry才能得到更準確的結果，且樹的數量和準確度並非成正比。

![](https://hackmd.io/_uploads/Hy3vQ6ZYn.png)

## 結論

此次嘗試了使用決策樹與隨機森林建造模型，分別有Decision Tree -with Gini、Decision Tree -with Gini (max_leaf_node = 12)、Decision Tree -with Entropy(max_leaf_node = 12)、Random Forest (mtry抽所有變數)、Random Forest (Best mtry=3) 與Random Forest (ntree=1000)六種方式。將此六種方式所建置的預測模型的Precision、Recall、F1與Accuracy分別列於下列圖表:

![](https://hackmd.io/_uploads/B1cqm6-Y3.png)

分別對照可知，準確率最高的為使用隨機森林並將Best mtry設為3的模型，F1最高的則為使用隨機森林並使用mtry抽所有變數的決策模型。從衡量結果來看，隨機森林所建造出來的模型基本上都比一般決策樹好，這個結果也符合理論。

後續想改進的地方為想先為資料集做class balance。由於此資料集的churn欄位，0的有8000筆，1的卻只有2000筆，比例為4:1，因此之後如果有機會改進的話想使用SMOTE方法人工創造一些資料，將兩者的資料量做平衡，以達更好的訓練成果。

![](https://hackmd.io/_uploads/SJan76bt3.png)
