import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 設定自訂義資料集
class CustomDataset(Dataset):
    # 設定初始函數
    def __init__(self, data_path, label_path):
        #　讀入訓練資料
        self.train_in = pd.read_csv(data_path)
        # 讀入訓練資料的標籤
        self.labels = pd.read_csv(label_path)
    
    # 設定get item函數
    def __getitem__(self, index):
        # 讀取特徵和標籤
        features = torch.tensor(self.train_in.iloc[index].values, dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[index].values, dtype=torch.long)
        return features, label
    
    # 設定len函數
    def __len__(self):
        return len(self.train_in)
    
    # 設定資料預處理函數
    def preprocess(self):
        # 將訓練資料中的'Serial No.'欄位移除
        self.train_in = self.train_in.drop(columns=['Serial No.'])
        # 將標籤中的'Serial No.'欄位移除
        self.labels = self.labels.drop(columns=['Serial No.'])

# 檢查是否有GPU可以使用
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 設定參數
input_size = 16
hidden_size =  514
hidden = torch.zeros(1, hidden_size)
# 有10個類別(0~9)
output_size = 10  
learning_rate = 0.0001
num_epochs = 1000
batch_size = 2
# 使用2層RNN
num_layers = 2  

# 設定自訂義資料集的路徑
train_data_path = r"D:/111-2/Foundation and Practice of Deep Learning/Online Character Recognition/triain_in.csv"
train_label_path = r"D:/111-2/Foundation and Practice of Deep Learning/Online Character Recognition/train_out.csv"
test_data_path = r"D:/111-2/Foundation and Practice of Deep Learning/Online Character Recognition/test_in.csv"

# 創建Dataset
train_dataset = CustomDataset(train_data_path, train_label_path)
test_dataset = CustomDataset(test_data_path, train_label_path)

# 進行資料預處理
train_dataset.preprocess()
test_dataset.preprocess()

# 創建Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 建立 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        # 設定隱藏狀態
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.1)
        self.fc1 = nn.Linear(hidden_size,256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    # 定義 Forward 函數
    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, _ = self.rnn(x, hidden)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.softmax(out)
        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        hidden = hidden.squeeze(0)  # 將多餘的維度去除
        return hidden.to(device)


# 初始化模型
model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.ASGD(model.parameters(), lr=learning_rate)

# 獲得現在時間(作為輸出的檔名)
now = datetime.now()
# 時間格式為 dd/mm/YY H:M:S
dt_string = now.strftime("%d.%m.%Y %H-%M-%S")

# 建立一個空的清單儲存loss(繪製圖表用)
losses = list()

# 訓練模型
for epoch in range(num_epochs):
    for features, labels in train_dataloader:
        
        features = features.float().to(device)
        labels = labels.squeeze().to(device)
        
        # 前向傳播
        outputs = model(features)
        
        # 計算損失
        loss = criterion(outputs, labels)
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 將每個 epoch 的 loss 記錄下來
    losses.append(loss.item())
    # 印出 loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

print('Finished Training')
print("date and time =", dt_string)

# 設定參數字典名稱
state_dict_name = dt_string + '.params'

# 儲存狀態字典
torch.save(model.state_dict(), state_dict_name)
print("Saved PyTorch Model State to " + state_dict_name)

# 將模型搬運至cpu準備做預測
device = 'cpu'
model = model.to(device)

# 使用訓練好的模型進行預測
predicted_labels = []
for features, _ in test_dataloader:
    features = features.float().to(device)
    
    # 前向傳播
    outputs = model(features)
    
    # 預測標籤，輸出為0-9各自的機率，因此找出機率最大的便為預測值
    predicted = torch.argmax(outputs, dim=1)
    predicted_labels.extend(predicted.tolist())

# 將預測結果寫入文件

# 讀入test data
submission = pd.read_csv(test_data_path)
# 設定第一個欄位
no = submission['Serial No.']
# 製作 Daraframe
submission = pd.DataFrame({'Serial No.':no,
                           'Label':predicted_labels})
# 製作 csv 檔
submission.to_csv(dt_string + "_submission.csv", index=False)
print(dt_string + "_submission.csv Saved Successfully")

# 將 loss 繪製出來
plt.plot(range(1, num_epochs+1), losses)
plt.xlabel('Epoch')  
plt.ylabel('Loss')
file_name = dt_string + '.png'
plt.savefig(file_name)
plt.show()