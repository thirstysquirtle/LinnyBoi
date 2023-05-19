import torch as t
import torch.nn as nn
import torch.nn.functional as F
import binance_functions as bn

device="cpu"

def f32t(data, device=device):
    return t.tensor(data, device=device, dtype=t.float32)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.network = nn.Sequential(nn.Linear(1,1))
        self.optimizer = t.optim.Adam(self.parameters(), lr=0.01)
        self.scheduler_vars = {"min_loss": float("inf"), "min_counter": 0}
        
    def forward(self, time):
        return self.network(f32t(time).unsqueeze(dim=1))
    
    def lr_scheduler(self, loss, patience=10, gamma=0.9, verbose=False):
        if loss < self.scheduler_vars["min_loss"]:
            self.scheduler_vars["min_loss"] = loss
            self.scheduler_vars["min_counter"] = 0
        else:
            self.scheduler_vars["min_counter"] = self.scheduler_vars["min_counter"] + 1
        
        if self.scheduler_vars["min_counter"] > patience:
            self.scheduler_vars["min_counter"] = 0
            # for loops changes learning rate of parameters
            new_lr = (self.optimizer.param_groups[0]["lr"] * gamma) + 1e-9
            self.optimizer.param_groups[0]["lr"] = new_lr
            if verbose:
                print(f"Learning Rate Adjusted to {new_lr}")

        return self.optimizer.param_groups[0]["lr"]
        

    def train_model(self, Xses, targets, num_epochs=1000 , loss_fn=F.mse_loss):
        Xses_proxy = [i for i in range(len(Xses))]
        loss_per_epoch = []
        print("debug " + str(Xses_proxy) )
        for i in range(num_epochs):
            preds = self.forward(Xses_proxy)
            loss = loss_fn(preds, targets)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.lr_scheduler(loss, patience=200)
            loss_per_epoch.append(loss.item())
            if self.optimizer.param_groups[0]["lr"] < 1e-6:
                print(f"i: {i}, learning_rate: {self.optimizer.param_groups[0]['lr']}")
                break   

        return loss_per_epoch

def get_targets(pandas_candles):
    return (f32t(pandas_candles[2] + pandas_candles[3])/2).unsqueeze(dim=1)
    
def get_future_prediction_xy(model, Xses):
    y = model([len(Xses)])
    length = len(Xses)
    x = (Xses[length - 1] - Xses[length - 2]) + Xses[length - 1]
    return [x, y]


model = Model().to(device)
candles = bn.get_candles("BNBBUSD", 24)
Xses = candles[0].values
# print(get_targets(candles))
# print(model(candles[0]))
# print()
losses = model.train_model(Xses, get_targets(candles), num_epochs=80000)
print(get_future_prediction_xy(model, Xses))

print(f"loss: {losses[len(losses)-1]}")
# print(model(Xses_proxy))
# print(model.optimizer.param_groups[0]["lr"])
print(Xses)
print(model([i for i in range(len(Xses))]))