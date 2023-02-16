import torch
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



class ClassificationLearner(object):
    def __init__(self, model, optimizer, criterion, lr_scheduler, dls_train, dls_valid, log_config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.dls_train = dls_train
        self.dls_valid = dls_valid
        
        self.log_config = log_config
        self.device = device

    def logger(self, outputs, phase):

        metrics = {
            'accuracy' : accuracy_score,
            'precision' : precision_score,
            'recall' : recall_score,
            'f1' : f1_score,
            'auroc' : roc_auc_score
        }

        preds = outputs['preds']
        probs = outputs['probs']
        labels = outputs['labels']
        loss = outputs['loss']

        if isinstance(log, str):
            log = [log]

        for l in self.log_config.log:
            panel = f"{phase}/{log}"

            if l == 'auroc':
                score = metrics[l](labels, probs, multi_class = self.log_config.multi_class, average = self.log_config.average)
            elif l == 'loss':
                wandb.log({panel : loss})
                continue
            elif l == 'accuracy':
                score = metrics[l](labels, preds)
            else:
                score = metrics[l](labels, preds, average = self.log_config.average, zero_division = 0)
            
            wandb.log({panel : score})
            

    def _fit(self):
        loss, accuracy = 0, 0
        self.model.train()
        for idx, batch in enumerate(self.dls_train):
            self.optimizer.zero_grad()
            
            batch_x, batch_y = batch.to(self.device), batch.activity.to(self.device)
            
            output = self.model.forward(batch_x)
            running_loss = self.criterion(output, batch_y)
            
            running_loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            loss += running_loss.item()
            
            running_acc = (output.argmax(dim = 1) == batch_y).sum()
            accuracy += running_acc.item()
        
        loss /= len(self.dls_train)
        accuracy /= len(self.dls_train.dataset)
        
        wandb.log({"train/loss" : loss})
        wandb.log({"train/accuracy" : accuracy})

    def _evaluate(self):
        self.model.eval()

        loss, accuracy = 0, 0
        probabilities = []
        predictions = []
        labels = []
        
        for idx, batch in enumerate(self.dls_valid):
            batch_x = batch.to(self.device)
            batch_y = batch.activity.cpu()
            
            with torch.no_grad():
                output = self.model.forward(batch_x).cpu()
                running_loss = self.criterion(output, batch_y)
                loss += running_loss.item()
                running_acc = (output.argmax(dim = 1) == batch_y).sum()
                accuracy += running_acc.item()

                probs = output.softmax(dim = 1).cpu().numpy()
                preds = output.argmax(dim = 1).cpu().numpy().reshape(-1)

                batch_y = batch_y.numpy().reshape(-1)

                predictions.append(preds)
                probabilities.append(probs)
                labels.append(batch_y)
                
                
        loss /= len(self.dls_valid)
        accuracy /= len(self.dls_valid.dataset)

        probabilities = np.vstack(probabilities)
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        outputs = {
            'preds' : predictions,
            'probs' : probabilities,
            'labels' : labels,
            'loss' : loss
        }

        if self.log_config is not None:
            self.logger(outputs, phase = "validation")
        else:
            wandb.log({"validation/loss" : loss})
            wandb.log({"validation/accuracy" : accuracy})

    def predict(self, dls_test, compute):
        self.model.eval()

        loss = []
        labels = []
        preds = []
        probs = []

        for idx, batch in enumerate(dls_test):
            batch_x = batch.to(self.device)
            batch_y = batch.activity.cpu()
            with torch.no_grad():
                output = self.model.forward(batch_x).cpu()
                loss.append(self.criterion(output, batch_y).item())
                preds.append(output.argmax(dim = 1).numpy())
                probs.append(output.softmax(dim = 1).numpy())
                labels.append(batch_y.numpy())

        probs = np.vstack(probs)
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        loss = np.array(loss)

        metrics = dict()

        metrics['Auroc'] = roc_auc_score(labels, probs, multi_class = self.log_config.multi_class, average = self.log_config.average)
        metrics['Accuracy'] = accuracy_score(labels, preds)
        metrics['Precision'] = precision_score(labels, preds, average = self.log_config.average, zero_division = 0)
        metrics['Recall'] = recall_score(labels, preds, average = self.log_config.average, zero_division = 0)
        metrics['F1'] = f1_score(labels, preds, average = self.log_config.average, zero_division = 0)

        print("Evaluation report: ")
        print("{:<15s} : {:.4f} ± {:.2f}".format("Loss", np.mean(loss), np.std(loss)))
        for k in metrics.keys():
            if k.lower() in compute:
                print("{:<15s} : {:.4f}".format(k, metrics[k]))

        return {"labels" : labels, "preds" : preds, "probs" : probs}
                
    def train(self, n_epochs):    
        for epoch in tqdm(range(n_epochs)):
            self._fit()
            self._evaluate()


class RegressionLearner(object):
    def __init__(self, model, optimizer, criterion, lr_scheduler, dls_train, dls_valid, log_config, device, mode):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.dls_train = dls_train
        self.dls_valid = dls_valid
        
        self.log_config = log_config
        self.device = device

    def logger(self, outputs, phase):

        metrics = {
            'mse' : mean_squared_error,
            'mae' : mean_absolute_error
        }

        logits = outputs['logits']
        labels = outputs['labels']
        loss = outputs['loss']

        if isinstance(log, str):
            log = [log]

        for l in self.log_comfig.log:
            panel = f"{phase}/{log}"
            
            if l == 'rmse':
                score = metrics[l](labels, logits, squared = False)
            else:
                score = l(labels, logits)
            
            wandb.log({panel : score})
            

    def _fit(self):
        loss = 0
        self.model.train()
        for idx, batch in enumerate(self.dls_train):
            self.optimizer.zero_grad()
            
            batch_x, batch_y = batch.to(self.device), batch.affinity.to(self.device)
            
            output = self.model.forward(batch_x)
            running_loss = self.criterion(output, batch_y)
            
            running_loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            loss += running_loss.item()
            
        loss /= len(self.dls_train)
        
        wandb.log({"train/loss" : loss})

    def _evaluate(self):
        self.model.eval()

        loss = 0
        logits = []
        labels = []
        
        for idx, batch in enumerate(self.dls_valid):
            batch_x = batch.to(self.device)
            batch_y = batch.affinity.cpu()
            
            with torch.no_grad():
                output = self.model.forward(batch_x).cpu()
                running_loss = self.criterion(output, batch_y)
                loss += running_loss.item()

                batch_y = batch_y.numpy().reshape(-1)

                logits.append(output)
                labels.append(batch_y)
                
                
        loss /= len(self.dls_valid)

        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        
        outputs = {
            'logits' : logits,
            'labels' : labels,
            'loss' : loss
        }

        if self.log_config is not None:
            self.logger(outputs, phase = "validation")
        else:
            wandb.log({"validation/loss" : loss})

    def predict(self, dls_test, compute):
        self.model.eval()

        loss = []
        labels = []
        logits = []

        for idx, batch in enumerate(dls_test):
            batch_x = batch.to(self.device)
            batch_y = batch.affinity.cpu()
            with torch.no_grad():
                output = self.model.forward(batch_x).cpu()
                loss.append(self.criterion(output, batch_y).item())
                logits.append(output)
                labels.append(batch_y.numpy())

        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        loss = np.array(loss)

        metrics = dict()

        metrics['MSE'] = mean_squared_error(labels, logits)
        metrics['RMSE'] = mean_squared_error(labels, logits, squared = True)
        metrics['MAE'] = mean_absolute_error(labels, logits)

        print("Evaluation report: ")
        print("{:<15s} : {:.4f} ± {:.2f}".format("Loss", np.mean(loss), np.std(loss)))
        for k in metrics.keys():
            if k.lower() in compute:
                print("{:<15s} : {:.4f}".format(k, metrics[k]))

        return {"labels" : labels, "logits" : logits}
                
    def train(self, n_epochs):    
        for epoch in tqdm(range(n_epochs)):
            self._fit()
            self._evaluate()