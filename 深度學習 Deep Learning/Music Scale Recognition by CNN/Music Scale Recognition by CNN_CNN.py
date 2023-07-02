import os
import pandas as pd

from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 製作自己的trainung dataset
class CustomImageDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir):
        
        # 讀入label
        self.img_labels = pd.read_table(annotations_file, header=None, sep=' ')
        
        # 設定image folder path
        self.img_dir = img_dir
    
    # 定義長度函數
    def __len__(self):
        return len(self.img_labels)

    # 定義get_item函數
    def __getitem__(self, idx):
        
        # 將image folder path與圖片名稱合併成圖片路徑
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # 使用pytorch提供的read_image函數，讀入image，並設為使用前三通道(忽略alpha通道)
        image = read_image(img_path)[:3, :, :]
        
        # 讀入圖片label
        label = self.img_labels.iloc[idx, 1]

        return image, label

# 讀入訓練資料集
train_data = CustomImageDataset(r'D:\111-2\Foundation and Practice of Deep Learning\Music Scale Recognition by CNN\data\truth.txt', 
                                r'D:\111-2\Foundation and Practice of Deep Learning\Music Scale Recognition by CNN\data\music_train')

# 設定訓練資料集的dataloader，並將batch size設置為64，並將資料洗牌
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

# 檢查是否有可用的GPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 建置神經網路
class Net(nn.Module):
    
    # 初始化
    def __init__(self):
        super(Net, self).__init__()

        # 宣告第一層捲積層
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)    
        
        # 池化層，使用MAX POOLING
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # 宣告第二層捲積層                    
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        
        # MLP第一層
        self.fc1 = nn.Linear(36195, 64)
        
        # MLP第二層，輸入為64，輸出為64
        self.fc2 = nn.Linear(64, 64)
        
        # MLP第三層，輸入為64，輸出為88(與要分類的類別數(0-87)一致)
        self.fc3 = nn.Linear(64, 88)
        
    def forward(self, x):
        
        # 第一層卷積層，結果丟入ReLU函數，再使用MaxPooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # 第二層卷積層，結果丟入ReLU函數，再使用MaxPooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # 將輸出平坦化，進入神經網路
        x = torch.flatten(x, 1)
       
        # 全連接層第一層 
        x = F.relu(self.fc1(x))
        
        # 全連接層第二層 
        x = F.relu(self.fc2(x))
        
        # 全連接層第三層 
        x = self.fc3(x)

        return x

# 宣告神經網路，並移至GPU
net = Net().to(device)

# 設定Loss funtion，使用cross entropy
criterion = nn.CrossEntropyLoss()

# 設定optimizer，使用SGD
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 45
batch_size = 64
learning_rate = 0.005
n_total_steps = len(train_dataloader)

# 開始訓練
for epoch in range(num_epochs):
    
    # 使用train data loader 遍歷所有訓練資料，並回傳每一batch的圖片及音階
    for i, (images, labels) in enumerate(train_dataloader):
        
        # 將圖片移至GPU
        images = images.to(device)
        
        # 將Label移至GPU
        labels = labels.to(device)
        
        # 初始化優化器
        optimizer.zero_grad()
        
        # 向前傳遞(Forward)
        outputs = net(images.type(torch.float32))
        loss = criterion(outputs, labels)
        
        # 向後傳遞(Backward)
        loss.backward()

        # 更新(Update)
        optimizer.step()

        # 印出此batch的loss
        print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.8f}')

print('Finished Training')

# 設定參數字典名稱
state_dict_name = 'CNN.params'

# 儲存狀態字典
torch.save(net.state_dict(), state_dict_name)
print("Saved PyTorch Model State to " + state_dict_name)

# 建立test data dataset
class TestDataset(Dataset):
    
    def __init__(self):
        
        # 設定image folder path
        self.img_dir = r'D:/111-2/Foundation and Practice of Deep Learning/Music Scale Recognition by CNN/data/music_test'
        
        # 讀入此path的圖片名稱字串
        self.file_names = os.listdir(self.img_dir)

    def __len__(self):
        return self.file_names.__len__()

    def __getitem__(self, idx):
        
        # 將image folder path與圖片名稱合併成圖片路徑
        img_path = os.path.join(self.img_dir, self.file_names[idx])
        
        # 使用pytorch提供的read_image函數，讀入image，並設為使用前三通道(忽略alpha通道)
        image = read_image(img_path)[:3, :, :]
        
        return image

# 宣告測試資料集
test_data = TestDataset()
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 初始化preds
preds = list()

# 將神經網路移至cpu
net = net.to('cpu')

# 用test_dataloader依序遍歷每一張圖片，並預測結果
for i, (images) in enumerate(test_dataloader):
    
    # 將圖片移至cpu
    images = images.to('cpu')

    # 預測結果
    outputs = net(images.type(torch.float32).to('cpu'))
    
    # 輸出為label 0~87 的各自機率，機率最大的就是所預測出來的label
    outputs = outputs.argmax(dim = 1)
    
    # 將預測結果存到preds中
    preds.extend(outputs.detach().numpy().tolist())

# 製作submission datafram
submission = pd.DataFrame({'filename':test_data.file_names,
                           'category':preds})

# 將submission存為csv檔
submission.to_csv('submission.csv', index=False)