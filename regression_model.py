import torch as t
import torch.nn as nn
import torch.nn.functional as F

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
        return self.network(time)
    
    def lr_scheduler(self, loss, patience=10, gamma=0.5):
        if loss < self.scheduler_vars["min_loss"]:
            self.scheduler_vars["min_loss"] = loss
            self.scheduler_vars["min_counter"] = 0
        else:
            self.scheduler_vars["min_counter"] = self.scheduler_vars["min_counter"] + 1
        
        if self.scheduler_vars["min_counter"] > patience:
            self.scheduler_vars["min_counter"] = 0
            # for loops changes learning rate of parameters
            new_lr = self.optimizer.param_groups[0]["lr"] * gamma
            self.optimizer.param_groups[0]["lr"] = new_lr
            print(f"Learning Rate Adjusted to {new_lr}")
        

    def train_model(self, preds, targets, num_epochs=1000 , loss_fn=F.mse_loss):
        loss_per_epoch = []
        for i in range(num_epochs):
            loss = loss_fn(preds, targets)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.lr_scheduler(loss)
            loss_per_epoch.append(loss.item())   


        return loss_per_epoch

model = Model().to(device)
