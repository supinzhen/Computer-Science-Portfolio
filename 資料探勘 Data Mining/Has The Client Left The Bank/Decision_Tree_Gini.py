#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd


# In[6]:


data_base = pd.read_csv('Bank_Customer.csv') #讀入資料庫


# In[7]:


data_base.info()


# In[8]:


data_base.head()


# # 資料處理

# In[9]:


import numpy as np


# In[10]:


feature_cols = ['credit_score','country','gender','age','tenure','balance','products_number','credit_card','active_member','estimated_salary']


# In[11]:


data = data_base.loc[:, feature_cols]


# In[12]:


data.head()


# In[13]:


data.info()


# In[14]:


target = data_base.churn.values


# In[15]:


target


# In[16]:


data_country = data['country']
data_gender = data['gender']
for i in range(0,10000):
    if(data_country[i] == 'France'):
        data_country[i] = 0
    elif(data_country[i] == 'Germany'):
        data_country[i] = 1
    elif(data_country[i] == 'Spain'):
        data_country[i] = 2
    if(data_gender[i] == 'Male'):
        data_gender[i] = 0
    elif(data_gender[i] == 'Female'):
        data_gender[i] = 1


# In[17]:


data.head()


# In[18]:


data['country'] = data['country'].astype(int)
data['gender'] = data['gender'].astype(int)


# In[19]:


data.info()


# # 訓練資料與測試資料

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.3,random_state = 42)


# In[22]:


x_train.shape


# In[23]:


y_train.shape


# In[24]:


x_test.shape


# In[25]:


y_test.shape


# # 建立決策樹

# In[26]:


from sklearn import tree


# In[27]:


ori_clf = tree.DecisionTreeClassifier(class_weight="balanced")
ori_clf = ori_clf.fit(x_train,y_train)


# In[28]:


tree.plot_tree(ori_clf)


# # 衡量模型

# In[29]:


from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[30]:


ori_predictions = ori_clf.predict(x_test) #用決策樹預測測試資料
ori_scores = cross_val_score(ori_clf,x_test,y_test, cv=10) #10-fold cross validation score
print("%0.3f accuracy with a standard deviation of %0.3f" % (ori_scores.mean(), ori_scores.std()))


# In[31]:


ori_cm = confusion_matrix(y_test,ori_predictions, labels=ori_clf.classes_) #混淆矩陣
ori_disp = ConfusionMatrixDisplay(confusion_matrix=ori_cm, display_labels=ori_clf.classes_)
ori_disp.plot()

plt.show()


# In[48]:


ori_accuracy = (ori_cm[0,0]+ori_cm[1,1])/3000
ori_recall = ori_cm[1,1]/(ori_cm[1,0]+ori_cm[1,1]) #計算recall
ori_precision = ori_cm[1,1]/(ori_cm[1,1]+ori_cm[0,1]) #計算precision
ori_F1 = 2 * (ori_precision * ori_recall) / (ori_precision + ori_recall) #計算F1
print("Accuracy: \t%0.3f" % (ori_accuracy))
print("Precision: \t%0.3f" % (ori_precision))
print("Recall: \t%0.3f" % (ori_recall))
print("F1 score: \t%0.3f" % (ori_F1))


# In[33]:


import graphviz 
ori_dot_data2 = tree.export_graphviz(ori_clf, out_file=None, 
                     feature_names=feature_cols,  
                     class_names=['0','1'],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(ori_dot_data2)  
graph.render("ori_tree_gini") 


# # 設置max_leaf_nodes

# In[47]:


for i in range (2,20) :
    test_my_clf = tree.DecisionTreeClassifier(splitter = "best",max_leaf_nodes= i)
    test_my_clf =  test_my_clf.fit(x_train,y_train)
    test_my_clf.score(x_test,y_test)
    scores = cross_val_score(test_my_clf,x_test,y_test, cv=10)
    predictions = test_my_clf.predict(x_test)
    cm = confusion_matrix(y_test, predictions, labels=test_my_clf.classes_)
    accuracy = (cm[0,0]+cm[1,1])/3000
    recall = cm[1,1]/(cm[1,0]+cm[1,1]) #計算recall
    precision = cm[1,1]/(cm[1,1]+cm[0,1]) #計算precision
    F1 = 2 * (precision * recall) / (precision + recall) #計算F1
    print("Accuracy: \t%0.3f" % (accuracy))
    print("Precision: \t%0.3f" % (precision))
    print("Recall: \t%0.3f" % (recall))
    print("F1 score: \t%0.3f" % (F1))
    print("%0.3f accuracy with a standard deviation of %0.3f, max_leaf_nodes = %i" % (scores.mean(), scores.std(),i))
    print("---------------------")


# In[35]:


my_clf = tree.DecisionTreeClassifier(splitter = "best",max_leaf_nodes = 12)
my_clf = my_clf.fit(x_train,y_train)
scores = cross_val_score(my_clf,x_test,y_test, cv=10)


# In[36]:


tree.plot_tree(my_clf)


# # 衡量新樹

# In[37]:


from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[49]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_clf,x_test,y_test, cv=10)
print("%0.3f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[39]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[40]:


predictions = my_clf.predict(x_test)
cm = confusion_matrix(y_test, predictions, labels=my_clf.classes_)
disp = ConfusionMatrixDisplay( confusion_matrix=cm , display_labels=my_clf.classes_)
disp.plot()

plt.show()


# In[44]:


accuracy = (cm[0,0]+cm[1,1])/3000
recall = cm[1,1]/(cm[1,0]+cm[1,1]) #計算recall
precision = cm[1,1]/(cm[1,1]+cm[0,1]) #計算precision
F1 = 2 * (precision * recall) / (precision + recall) #計算F1
print("Accuracy: \t%0.3f" % (accuracy))
print("Precision: \t%0.3f" % (precision))
print("Recall: \t%0.3f" % (recall))
print("F1 score: \t%0.3f" % (F1))


# In[45]:


import graphviz 
dot_data = tree.export_graphviz(my_clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("my_tree_gini") 


# In[46]:


dot_data2 = tree.export_graphviz(my_clf, out_file=None, 
                     feature_names=feature_cols,  
                     class_names=['0','1'],  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data2)  
graph.render("my_tree2_gini") 


# In[ ]:





# In[ ]:





# In[ ]:




