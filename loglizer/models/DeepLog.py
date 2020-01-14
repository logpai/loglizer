import torch
import math
import torch.optim as optim
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import defaultdict

class DeepLog(nn.Module):
    def __init__(self, num_labels, hidden_size=100, num_directions=2, topk=9, device="cpu"):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.topk = topk
        self.device = self.set_device(device)
        self.rnn = nn.LSTM(input_size=1, hidden_size=self.hidden_size, batch_first=True, bidirectional=(self.num_directions==2))
        self.criterion = nn.CrossEntropyLoss()
        self.prediction_layer = nn.Linear(self.hidden_size * self.num_directions, num_labels + 1)

    def forward(self, input_dict):
        y = input_dict["window_y"].long().view(-1).to(self.device)
        self.batch_size = y.size()[0]
        x = input_dict["x"].view(self.batch_size, -1, 1).to(self.device)
        outputs, hidden = self.rnn(x.float(), self.init_hidden())
        logits = self.prediction_layer(outputs[:,-1,:])
        y_pred = logits.softmax(dim=-1)
        loss = self.criterion(logits, y)
        return_dict = {'loss': loss, 'y_pred': y_pred}
        return return_dict

    def set_device(self, gpu=-1):
        if gpu != -1 and torch.cuda.is_available():
            device = torch.device('cuda: ' + str(gpu))
        else:
            device = torch.device('cpu')   
        return device

    def init_hidden(self):
        h0 = torch.zeros(self.num_directions, self.batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_directions, self.batch_size, self.hidden_size).to(self.device)
        return (h0, c0)

    def fit(self, train_loader, epoches=10):
        self.to(self.device)
        model = self.train()
        optimizer = optim.Adam(model.parameters())
        for epoch in range(epoches):
            batch_cnt = 0
            epoch_loss = 0
            for batch_input in train_loader:
                loss = model.forward(batch_input)["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt
            print("Epoch {}/{}, training loss: {:.5f}".format(epoch+1, epoches, epoch_loss))
    
    def evaluate(self, test_loader):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            for batch_input in test_loader:
                return_dict = self.forward(batch_input)
                y_pred = return_dict["y_pred"]

                store_dict["SessionId"].extend(batch_input["SessionId"].data.cpu().numpy().reshape(-1))
                store_dict["y"].extend(batch_input["y"].data.cpu().numpy().reshape(-1))
                store_dict["window_y"].extend(batch_input["window_y"].data.cpu().numpy().reshape(-1))
                window_prob, window_pred = torch.max(y_pred, 1)
                store_dict["window_pred"].extend(window_pred.data.cpu().numpy().reshape(-1))
                store_dict["window_prob"].extend(window_prob.data.cpu().numpy().reshape(-1))
                top_indice = torch.topk(y_pred, self.topk)[1] # b x topk
                store_dict["topk_indice"].extend(top_indice.data.cpu().numpy())

            window_pred = store_dict["window_pred"]
            window_y = store_dict["window_y"]
            
            store_df = pd.DataFrame(store_dict)
            store_df["anomaly"] = store_df.apply(lambda x: x["window_y"] not in x["topk_indice"], axis=1).astype(int)

            store_df.drop(["window_pred", "window_y"], axis=1)
            store_df = store_df.groupby('SessionId', as_index=False).sum()
            store_df["anomaly"] = (store_df["anomaly"] > 0).astype(int)
            store_df["y"] = (store_df["y"] > 0).astype(int)
            y_pred = store_df["anomaly"]
            y_true = store_df["y"]

            metrics = {"window_acc" : accuracy_score(window_y, window_pred),
            "session_acc" : accuracy_score(y_true, y_pred),
            "f1" : f1_score(y_true, y_pred),
            "recall" : recall_score(y_true, y_pred),
            "precision" : precision_score(y_true, y_pred)}
            print([(k, round(v, 5))for k,v in metrics.items()])
            return metrics