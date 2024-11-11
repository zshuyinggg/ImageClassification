### YOUR CODE HERE
import torch
import os, time
import numpy as np
from Network import *
from ImageUtils import parse_record
import torch.optim as optim
from tqdm import tqdm
from Network import SimpleDLA
"""This script defines the training, validation and testing process.
"""
class MyModel(object):

 
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config

        self.network=SimpleDLA().cuda()
        self.lr=self.config['lr']
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.loss=nn.CrossEntropyLoss()
        self.optimizer=optim.SGD(self.network.parameters(), lr=self.lr, weight_decay = self.config['weight_decay'], momentum = 0.9)
        ### YOUR CODE HERE
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def train(self, x_train, y_train, config, x_valid, y_valid):
        # Determine how many batches in an epoch
        num_train_samples,num_val_samples = x_train.shape[0],x_valid.shape[0]
        num_train_batches = num_train_samples // self.config['batch_size']
        num_val_batches = num_val_samples // self.config['batch_size']

        print('### Training... ###')
        best_acc=0
        for epoch in range(1, config['max_epoch']+1):
            self.network.train()

            adjust_learning_rate(config,self.optimizer,epoch)
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_train_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            curr_x_train=np.stack([parse_record(record,training=True) for record in curr_x_train],axis=0)
            x_valid=np.stack([parse_record(record,training=False) for record in x_valid],axis=0)
            

            
            for i in range(num_train_batches):
                curr_batch_x=torch.tensor(curr_x_train[self.config['batch_size']*i:self.config['batch_size']*(i+1)]).cuda().float()
                curr_batch_y=torch.tensor(curr_y_train[self.config['batch_size']*i:self.config['batch_size']*(i+1)]).cuda().long()
                self.optimizer.zero_grad()
                predicted_y=self.network(curr_batch_x)
                loss=self.loss(predicted_y,curr_batch_y)
                
                loss.backward()
                self.optimizer.step()

                # print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_train_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))
            curr_acc=self.evaluate(curr_x_train,curr_y_train)

            curr_acc=self.evaluate(x_valid,y_valid)

            if curr_acc>best_acc:
                best_acc=curr_acc
                flag=1
                self.save(epoch,best_acc)
            # else:
            #     flag-=1
            #     if flag==-30:
            #         print('validation acc did not improve for 10 epochs. stop training.')
            #         break
            self.scheduler.step()
    def evaluate(self,x,y,s='validation'):
        self.network.eval()
        x=np.stack([parse_record(record,training=False) for record in x],axis=0)
        preds = []
        with torch.no_grad():
            for i in tqdm(range(x.shape[0]//self.config['batch_size'])):
                torch.cuda.empty_cache()
                curr_batch_x=torch.tensor(x[self.config['batch_size']*i:self.config['batch_size']*(i+1)]).cuda().float()
                predicted_y=self.network(curr_batch_x)
                preds.append(predicted_y)
                del curr_batch_x,predicted_y
            # last batch 
            if x.shape[0]%self.config['batch_size']!=0:
                torch.cuda.empty_cache()

                curr_batch_x=torch.tensor(x[-(x.shape[0]%self.config['batch_size']):]).cuda().float()
                predicted_y=self.network(curr_batch_x)
                preds.append(predicted_y)         
                del curr_batch_x,predicted_y



        y = torch.tensor(y)
        preds = torch.argmax(torch.vstack(preds),dim=1).cpu().detach()
        acc=torch.sum(preds==y)/y.shape[0]
        print(s+' accuracy: {:.4f}'.format(acc))
        return acc


    
    def save(self, epoch,best_acc):
        checkpoint_path = os.path.join(self.config['save_dir'], '%s-%d-%.2f.ckpt'%(self.config['model_name'],epoch,best_acc))
        os.makedirs(self.config['save_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


    def predict_prob(self, x):
        self.network.eval()
        x=np.stack([parse_record(record,training=False) for record in x],axis=0)
        preds = []
        with torch.no_grad():
            for i in tqdm(range(x.shape[0]//self.config['batch_size'])):
                curr_batch_x=torch.tensor(x[self.config['batch_size']*i:self.config['batch_size']*(i+1)]).cuda().float()
                predicted_y=self.network(curr_batch_x)
                preds.append(predicted_y)
                del curr_batch_x,predicted_y


            if x.shape[0]%self.config['batch_size']!=0:
                torch.cuda.empty_cache()
                curr_batch_x=torch.tensor(x[-(x.shape[0]%self.config['batch_size']):]).cuda().float()
                predicted_y=self.network(curr_batch_x)
                preds.append(predicted_y)         
                del curr_batch_x,predicted_y


        preds_probs=torch.vstack(preds).cpu().numpy().reshape((2000,10))
        return preds_probs


def adjust_learning_rate(config, optimizer, epoch):
    if config['lr_adjust'] == 'step':
        """Sets the learning rate to the initial LR decayed by 10
        every 30 epochs"""
        lr = config['lr'] * (config['step_ratio'] ** (epoch // 30))
    else:
        raise ValueError()
    # print('Epoch [{}] Learning rate: {:0.6f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

