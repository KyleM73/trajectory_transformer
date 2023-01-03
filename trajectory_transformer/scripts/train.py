import torch
import matplotlib.pyplot as plt

def train_loop(dataloader, model, src_mask, tgt_mask, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X, y, src_mask, tgt_mask)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if batch % 100 != 0:
        print(f"loss: {loss:>7f}  [{size:>5d}/{size:>5d}]")

    return losses

def test_loop(dataloader, model, src_mask, tgt_mask, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X, y, src_mask, tgt_mask)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return [test_loss]

def save_outputs(t,loss,test_loss,model,save_dir):
    fig,ax = plt.subplots()
    t_loss = [tt*(t+1)/len(loss) for tt in range(len(loss))]
    t_test_loss = [tt+1 for tt in range(len(test_loss))]
    ax.plot(t_loss,loss,label="loss")
    ax.plot(t_test_loss,test_loss,label="test_loss")
    ax.legend()
    fig.savefig(save_dir+"/loss.png")
    plt.close(fig)
    torch.save(model.state_dict(),save_dir+"/model{}.pt".format(t+1))

class NoamOpt:
    """
    Optim wrapper that implements rate.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """
        Update parameters and rate.
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        """
        Implement `lrate` above.
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))