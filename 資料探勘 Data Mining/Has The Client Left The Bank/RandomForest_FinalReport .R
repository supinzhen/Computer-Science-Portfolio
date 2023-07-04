#control + Enter 執行單行指令

#package
library(randomForest)
library(vita)
library(caret)
library(mlbench)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(cowplot)
library(skimr)

#read csv file
url <- "C:/Users/xsfb1/Documents/Bank_Customer.csv"
data <- read.csv(url, header=TRUE) 
str(data) #show data info

#check data
sum(duplicated(data$customer_id))#check repeat data
data[data=="?"] #is data include "?"
data[data=="NA"] #is data include "NA"
skim(data) #更仔細的資料簡介
sum(data$gender=="Male")/sum(data$gender=="Female")*100 #計算男女比

dd <- data[,-1] #將customer_id拿掉不納入計算

#draw data
ggplot(data = dd) + geom_bar(mapping = aes(x = credit_card)) #持卡柱狀圖

#年齡柱狀+折線圖
ggplot(dd, aes(credit_score)) + geom_line()

ggplot(data = dd)+(aes(x = tenure)) + 
  geom_histogram(bins = 30, aes(y = after_stat(density)), alpha = 0.5) +
  geom_density()

#資料判斷
data[data$tenure=="0"&data$churn=="1",] #條件:使用期為0且為流失顧客條列所有資料
table(data$tenure=="0"&data$churn=="1") #條件:使用期為0且為流失顧客條列總數
table(data$tenure=="0"&data$churn=="0"&data$balance=="0")#條件:使用期為0且為流失顧客且餘額為0條列總數

#describe question
#顧客流失和使用期柱狀圖
ggplot(data = dd,mapping = aes(x = tenure)) +  
  geom_bar(aes(group = churn,fill = churn)) +
  ggtitle("tenure and churn ")

#顧客流失和餘額柱狀圖
 ggplot(data = dd,mapping = aes(x =churn )) +  
  geom_bar(aes(group = balance,fill = balance)) +
  ggtitle("balance and churn ")

#顧客流失柱狀圖
 ggplot(data = dd,mapping = aes(x = churn)) +  
  geom_bar(aes(group = churn)) +
  ggtitle("churn ")

#性別和顧客流失柱狀圖
ggplot(data = dd,mapping = aes(x = gender)) + 
  geom_bar(aes(group = churn,fill = churn)) +
  ggtitle("gender and churn ")

#產品編號和顧客流失柱狀圖
ggplot(data = dd,mapping = aes(x = products_number)) +  
  geom_bar(aes(group = churn,fill = churn)) +
  ggtitle("products_number and churn ")

#活動會員和顧客流失柱狀圖
 ggplot(data = dd,mapping = aes(x = active_member)) + 
  geom_bar(aes(group = churn,fill = churn)) +
  ggtitle("active_member and churn ")

 #年齡和顧客流失柱狀圖
 ggplot(data = dd,mapping = aes(x = age)) + 
   geom_bar(aes(group = churn,fill = churn)) +
   ggtitle("age and churn ")
 
#Random Forest
set.seed(42) #To randomly sampling things

dd$churn <- as.factor(dd$churn) # Now convert to a factor
class(dd$churn)

#資料預處理
index.train = sample(1:nrow(dd), size=ceiling(0.7*nrow(dd))) # training dataset index
train = dd[index.train, ] # trainind data
test = dd[-index.train, ] # test data

#random forest
#original
model <- randomForest(churn ~ ., data=train, 
                      proximity=TRUE,   #估計樣本間的相似
                      mtry =ncol(dd)-1, #決定抽取多少個變數去建每棵決策樹
                      importance=TRUE)  #结合importance()函数使用，用來觀察重要變數
model

pre.rf <- predict (model,newdata = test)# Perform on the test set
#confusionMatrix
#Pos Pred Value = Precision
#Sensitivity = Recall
model_con<-confusionMatrix(pre.rf, test$ churn)
model_con
# calculating F1
F1 <- (2 * model_con$byClass[3]* model_con$byClass[1]) / ( model_con$byClass[3] + model_con$byClass[1])
print(F1)

#get best mtry
tuneRF(x=train[,-11], y=train[,11], 
       ntree = 500) 

#change mtry
model.three <- randomForest(churn ~ ., data=train_re, 
                      proximity=TRUE,   #估計樣本間的相似
                      mtry =3, #決定抽取多少個變數去建每棵決策樹
                      importance=TRUE)  #结合importance()函数使用，用來觀察重要變數
model.three

pre.rf.three <- predict (model.three, test)# Perform on the test set
#confusionMatrix
three_con<-confusionMatrix(pre.rf.three, test$ churn)
three_con
# calculating F1
F1 <- (2 * three_con$byClass[3]* three_con$byClass[1]) / ( three_con$byClass[3] + three_con$byClass[1])
print(F1)

#查看importance有哪幾個
importance(model.three)
varImpPlot (model.three, sort = TRUE)

#假設設定1000棵tree
model.onet <- randomForest(churn ~ ., data=test, 
                           ntree=1000,     #ntree=how many decision tree was generate
                           proximity=TRUE,   #估計樣本間的相似
                           mtry =3, #決定抽取多少個變數去建每棵決策樹
                           importance=TRUE)  #结合importance()函数使用，用來觀察重要變數
model.onet

pre.rf.onet <- predict (model.onet,newdata = test)# Perform on the test set
onet_con<-confusionMatrix(pre.rf.onet, test$ churn)
onet_con
F1 <- (2 * onet_con$byClass[3]* onet_con$byClass[1]) / ( onet_con$byClass[3] + onet_con$byClass[1])
print(F1)


#存oob.error.data
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model.three$err.rate), times=3),
  Type=rep(c("OOB", "0","1" ), each=nrow(model.three$err.rate)),
  Error=c(model.three$err.rate[,"OOB"], 
          model.three$err.rate[,"0"],
          model.three$err.rate[,"1"] ))

#show oob.values
oob.values <- vector(length=10)
for(i in 1:10) {
  oob.values[i] <- model.three$err.rate[nrow(model.three$err.rate),1]
}
oob.values

oob.error.data <- data.frame(
  Trees=rep(1:nrow(model.onet$err.rate), times=3),
  Type=rep(c("OOB", "0", "1"), each=nrow(model.onet$err.rate)),
  Error=c(model.onet$err.rate[,"OOB"], 
          model.onet$err.rate[,"0"], 
          model.onet$err.rate[,"1"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

pre.rf.onet <- predict (model.onet,newdata = test)# Perform on the test set

#draw
rpart.plot(Tree_model2_cv$finalModel, extra=4)
plot.train(Tree_model2_cv)
