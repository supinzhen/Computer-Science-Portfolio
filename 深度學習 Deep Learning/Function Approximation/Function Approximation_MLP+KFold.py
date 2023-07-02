import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np

# 製作data module
class FunctionApproximation(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        # 初始化
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            # 讀入訓練原始資料
            self.raw_train = pd.read_csv('train.csv')
            # 讀入測試原始資料
            self.raw_val = pd.read_csv('test.csv')     
    
    # 宣告資料與處理函數
    def preprocess(self):
        # 將id與y去除
        features = pd.concat((self.raw_train.drop(columns=['id', 'y']),self.raw_val.drop(columns=['id'])))
        # 尋找非數值的資料欄位
        numeric_features = features.dtypes[features.dtypes!='object'].index
        # 將缺失的資料補0
        features[numeric_features] = features[numeric_features].fillna(0)
        # 將預處理後的資料設為訓練資料
        self.train = features[:self.raw_train.shape[0]].copy()
        # 將預'y'設為目標
        self.train['y'] = self.raw_train['y']
        self.val = features[self.raw_train.shape[0]:].copy()
          
    def get_dataloader(self, train):
        # 在訓練階段，data為self.train，在測試階段，data為self.val
        data = self.train if train else self.val
        if 'y' not in data: return
        get_tensor = lambda x: torch.tensor(x.values, dtype=torch.float32, device="cuda")
        # 製作x與y
        tensors = (get_tensor(data.drop(columns=['y'])),  # X
                torch.log(get_tensor(data['y'])).reshape((-1, 1)))  # Y
        return self.get_tensorloader(tensors, train)
    
# 實作MLP
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2, num_hiddens_3, 
                                    num_hiddens_4, num_hiddens_5, num_hiddens_6,
                                    dropout_1, dropout_2, dropout_3, dropout_4, dropout_5, lr):
        # 初始化
        super().__init__()
        self.save_hyperparameters()
        # 設定各層
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), nn.Dropout(dropout_1), 
            nn.LazyLinear(num_hiddens_2), nn.ReLU(), nn.Dropout(dropout_2), 
            nn.LazyLinear(num_hiddens_3), nn.ReLU(), nn.Dropout(dropout_3), 
            nn.LazyLinear(num_hiddens_4), nn.Dropout(dropout_4), 
            nn.LazyLinear(num_hiddens_5), nn.ReLU(),nn.Dropout(dropout_5), 
            nn.LazyLinear(num_hiddens_6), nn.ReLU(),
            nn.LazyLinear(num_outputs))
        
    # 設定損失函數
    def loss(self, y_hat, y):
        # 使用Mean Square Error
        fn = nn.MSELoss() 
        mse = fn(y_hat, y)        
        print(mse.item())
        return mse

    # 設定optimizer
    def configure_optimizers(self):
        # 使用 stochastic gradient descent
        return torch.optim.SGD(self.parameters(), self.lr)

# 實作K-Fold
def k_fold_data(data, k):
    rets = []
    # 將data分成k分
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(FunctionApproximation(data.batch_size, data.train.drop(index=idx), data.train.loc[idx]))
    return rets

data = FunctionApproximation(batch_size=256)
# 資料預處理
data.preprocess()
 
hparams = {'num_outputs':1, 'num_hiddens_1':2048, 'num_hiddens_2':2048, 
           'num_hiddens_3':1024, 'num_hiddens_4':1024, 'num_hiddens_5':512, 'num_hiddens_6':256,
           'dropout_1':0.03, 'dropout_2':0.03, 'dropout_3':0.02,'dropout_4':0.02, 'dropout_5':0.02, 'lr':0.05}

def k_fold(trainer, data, k):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        # 宣告模型，並移到GPU
        model = DropoutMLP(**hparams).to("cuda")
        # 讀入訓練好的狀態字典
        model.load_state_dict(torch.load('mlpv.params'))
        if i != 0: model.board.display = False
        # 訓練模型
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        # 儲存狀態字典
        torch.save(model.state_dict(), 'mlpv.params')
        # 儲存模型
        models.append(model)
    print(f'average validation loss = {sum(val_loss)/len(val_loss)}')
    return models

# 宣告Trainer
trainer = d2l.Trainer(max_epochs = 500)
# 宣告模型
models = k_fold(trainer, data, k = 5)

# 預測test data
preds = [model(torch.tensor(data.val.values, dtype=torch.float32, device="cuda")).to('cpu')
         for model in models]
ensemble_preds = torch.exp(torch.cat(preds, 1)).mean(1)
ID = torch.arange(1, 2001)
# 製作submission dataframe
submission = pd.DataFrame({'id':ID.detach().numpy(),
                           'y':ensemble_preds.detach().numpy()})
# 儲存submission dataframe 到一個csv file
submission.to_csv('submissionkfold.csv', index=False)