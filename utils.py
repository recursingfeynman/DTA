import torch
import numpy as np
import wandb
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

        if isinstance(self.log_config.log, str):
            self.log_config.log = [self.log_config.log]

        for l in self.log_config.log:
            panel = f"{phase}/{l}"

            if l == 'auroc':
                if self.log_config.multi_class == 'raise':
                    probs = probs[:, 1]
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
        metrics['AUROC'] = roc_auc_score
        metrics['Accuracy'] = accuracy_score
        metrics['Precision'] = precision_score
        metrics['Recall'] = recall_score
        metrics['F1'] = f1_score

        print("Evaluation report: ")
        print("{:<15s} : {:.4f} ± {:.2f}".format("Loss", np.mean(loss), np.std(loss)))

        if isinstance(compute, str):
            compute = [compute]

        for l in compute:
            if l.lower() == 'auroc':
                if self.log_config.multi_class == 'raise':
                    probs = probs[:, 1]
                score = metrics["AUROC"](labels, probs, multi_class = self.log_config.multi_class, average = self.log_config.average)
            else:
                score = metrics[l](labels, preds, average = self.log_config.average, zero_division = 0)

            print("{:<15s} : {:.4f}".format(l, score))

        return {"labels" : labels, "preds" : preds, "probs" : probs}
                
    def train(self, n_epochs):    
        for epoch in tqdm(range(n_epochs)):
            self._fit()
            self._evaluate()


class RegressionLearner(object):
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
            'mse' : mean_squared_error,
            'mae' : mean_absolute_error,
            'r2' : r2_score
        }

        logits = outputs['logits']
        labels = outputs['labels']
        loss = outputs['loss']

        if isinstance(self.log_config.log, str):
            self.log_config.log = [self.log_config.log]

        for l in self.log_config.log:
            panel = f"{phase}/{l}"
            
            if l == 'rmse':
                score = metrics[l](labels, logits, squared = False)
            elif l == 'loss':
                wandb.log({panel : loss})
                continue
            else:
                score = metrics[l](labels, logits)
            
            wandb.log({panel : score})
            

    def _fit(self):
        loss = 0
        self.model.train()
        for idx, batch in enumerate(self.dls_train):
            self.optimizer.zero_grad()
            
            batch_x, batch_y = batch.to(self.device), batch.affinity.to(self.device)
            
            output = self.model.forward(batch_x).flatten()
            running_loss = self.criterion(output, batch_y)
            
            running_loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            self.optimizer.step()

            if self.lr_scheduler is not None and 'OneCycleLR' in str(self.lr_scheduler.__class__):
                self.lr_scheduler.step()
            
            loss += running_loss.item()
            
        loss /= len(self.dls_train)
        
        wandb.log({"train/loss" : loss})

        if self.lr_scheduler is not None and 'OneCycleLR' not in str(self.lr_scheduler.__class__):
            self.lr_scheduler.step()

    def _evaluate(self):
        self.model.eval()

        loss = 0
        logits = []
        labels = []
        
        for idx, batch in enumerate(self.dls_valid):
            batch_x = batch.to(self.device)
            batch_y = batch.affinity.cpu()
            
            with torch.no_grad():
                output = self.model.forward(batch_x).flatten().cpu()
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

    def predict(self, dls_test, compute_metrics):
        self.model.eval()

        loss = []
        labels = []
        logits = []

        for idx, batch in enumerate(dls_test):
            batch_x = batch.to(self.device)
            batch_y = batch.affinity.cpu()
            with torch.no_grad():
                output = self.model.forward(batch_x).flatten().cpu()
                loss.append(self.criterion(output, batch_y).item())
                logits.append(output)
                labels.append(batch_y.numpy())

        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        loss = np.array(loss)

        metrics = dict()

        metrics['mse'] = mean_squared_error
        metrics['rmse'] = mean_squared_error
        metrics['mae'] = mean_absolute_error
        metrics['r2'] = r2_score

        print("Evaluation report: ")
        print("{:<15s} : {:.4f} ± {:.2f}".format("Loss", np.mean(loss), np.std(loss)))

        if isinstance(compute_metrics, str):
            compute_metrics = [compute_metrics]

        for l in compute_metrics:
            if l.lower() == "rmse":
                score = metrics[l](labels, logits, squared = False)
            else:
                score = metrics[l](labels, logits)

            print("{:<10s} : {:.4f}".format(l.upper(), score))         

        return labels, logits
                
    def train(self, n_epochs):    
        for epoch in tqdm(range(n_epochs)):
            self._fit()
            self._evaluate()

def compute_embeddings(model, loader, device):
    '''Сalculates the embeddings using the proposed model'''

    model.eval()

    embeddings = []
    labels = []

    for batch in loader:
        batch_x = batch.to(device)
        batch_y = batch.activity.to(device)
        # batch_x, batch_y = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model.forward(batch_x)
            embeddings.append(output)
            labels.append(batch_y)
        
    embeddings = torch.cat(embeddings, dim = 0).cpu()
    labels = torch.cat(labels, dim = 0).cpu()

    return embeddings, labels

def compute_mean_embedding_per_class(embeddings, labels):
    '''Function that computes mean embedding of each class'''

    mean_embeddings = []
    mean_embeddings_class = []

    for label in labels.unique():
        mask = (labels == label)
        class_embeddings = embeddings[mask]
        mean_embeddings.append(class_embeddings.mean(0))
        mean_embeddings_class.append(label)

    mean_embeddings = torch.stack(mean_embeddings)
    mean_embeddings_class = torch.tensor(mean_embeddings_class)
        
    return mean_embeddings, mean_embeddings_class

def predict_class(embeddings, reference, classes):
    '''Finds class with minimum cosine distance for each embeddings'''
    
    distance = 1 - pairwise_cosine(embeddings, reference, zero_diag = False)
    nearest_index = torch.argmin(distance, dim = 1)
    predicted_class = classes[nearest_index]

    return predicted_class

def pairwise_cosine(x, y = None, zero_diag = True):
    '''Compute cosine similarity pairwise'''
    
    if y is None:
        y = x.clone()
        
    x_norm = torch.linalg.norm(x, ord = 2, dim = 1)
    x = x.div(x_norm.unsqueeze(1))
    y_norm = torch.linalg.norm(y, ord = 2, dim = 1)
    y = y.div(y_norm.unsqueeze(1))
    
    cosine = x @ y.T
    
    if zero_diag:
        cosine.fill_diagonal_(0)
    
    return cosine


class TripletLearner(object):
    def __init__(self, model, optimizer, criterion, lr_scheduler, dls_train, dls_valid, log_config, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.dls_train = dls_train
        self.dls_valid = dls_valid
        
        self.log_config = log_config
        self.device = device

        self.reference_emb = None
        self.reference_class = None

    def logger(self, outputs, phase):

        metrics = {
            'accuracy' : accuracy_score,
            'precision' : precision_score,
            'recall' : recall_score,
            'f1' : f1_score,
            'auroc' : roc_auc_score
        }

        preds = outputs['preds']
        labels = outputs['labels']
        loss = outputs['loss']

        if isinstance(self.log_config.log, str):
            self.log_config.log = [self.log_config.log]

        for l in self.log_config.log:
            panel = f"{phase}/{l}"

            if l == 'auroc':
                continue
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

        self.model.eval()
        with torch.no_grad():
            tr_embeddings, tr_labels = compute_embeddings(self.model, self.dls_train, self.device)
            self.reference_emb, self.reference_class = compute_mean_embedding_per_class(tr_embeddings, tr_labels)

        self.model.train()
        for idx, batch in enumerate(self.dls_train):
            self.optimizer.zero_grad()
            
            batch_x, batch_y = batch.to(self.device), batch.activity.to(self.device)
            # batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
            embeddings = self.model.forward(batch_x)
            running_loss = self.criterion(embeddings, batch_y)
            
            running_loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            loss += running_loss.item()
            
            predicted_class = predict_class(embeddings.detach().cpu(), self.reference_emb, self.reference_class).to(self.device)
            running_acc = (predicted_class == batch_y).sum()
            accuracy += running_acc.item()
        
        loss /= len(self.dls_train)
        accuracy /= len(self.dls_train.dataset)
        if self.log_config is not None:
            wandb.log({"train/loss" : loss})
            wandb.log({"train/accuracy" : accuracy})
            return loss, accuracy
        else:
            return loss, accuracy

    def _evaluate(self):
        self.model.eval()

        loss, accuracy = 0, 0
        embeddings = []
        labels = []
        
        for idx, batch in enumerate(self.dls_valid):
            batch_x = batch.to(self.device)
            batch_y = batch.activity.cpu()
            # batch_x, batch_y = batch[0].to(self.device), batch[1].to(self.device)
            
            with torch.no_grad():
                emb = self.model.forward(batch_x).cpu()
                running_loss = self.criterion(emb, batch_y)
                loss += running_loss.item()
                batch_y = batch_y.flatten()
                embeddings.append(emb)
                labels.append(batch_y)
                
                
        loss /= len(self.dls_valid)

        embeddings = torch.cat(embeddings, dim = 0)
        labels = torch.cat(labels, dim = 0)
        predictions = predict_class(embeddings, self.reference_emb, self.reference_class)
        accuracy = (predictions == labels).sum().item() / len(self.dls_valid.dataset)
        
        outputs = {
            'preds' : predictions.numpy(),
            'labels' : labels.numpy(),
            'loss' : loss
        }

        if self.log_config is not None:
            self.logger(outputs, phase = "validation")
            return loss, accuracy
        else:
            return loss, accuracy

    def predict(self, dls_test, compute, multi_class = True, average = "macro"):
        self.model.eval()

        loss = []
        labels = []
        embeddings = []

        for idx, batch in enumerate(dls_test):
            batch_x = batch.to(self.device)
            batch_y = batch.activity.cpu()
            with torch.no_grad():
                emb = self.model.forward(batch_x).cpu()
                loss.append(self.criterion(emb, batch_y).item())
                labels.append(batch_y.flatten())
                embeddings.append(emb)

        labels = torch.cat(labels, dim = 0).numpy()
        embeddings = torch.cat(embeddings, dim = 0)
        predictions = predict_class(embeddings, self.reference_emb, self.reference_class).numpy()

        metrics = dict()
        metrics['AUROC'] = roc_auc_score
        metrics['Accuracy'] = accuracy_score
        metrics['Precision'] = precision_score
        metrics['Recall'] = recall_score
        metrics['F1'] = f1_score

        print("Evaluation report: ")
        print("{:<15s} : {:.4f} ± {:.2f}".format("Loss", np.mean(loss), np.std(loss)))

        if isinstance(compute, str):
            compute = [compute]

        for l in compute:
            if l.lower() == 'auroc':
                if not multi_class:
                    probs = probs[:, 1]
                    score = metrics["AUROC"](labels, probs, average = average)
                else:
                    score = metrics["AUROC"](labels, probs, multi_class = "ovo", average = average)
            else:
                score = metrics[l](labels, predictions, average = average, zero_division = 0)

            print("{:<15s} : {:.4f}".format(l, score))

        return {"labels" : labels, "preds" : predictions, "emb" : embeddings}
                
    def train(self, n_epochs):    
        for epoch in tqdm(range(n_epochs)):
            tr_loss, tr_acc = self._fit()
            val_loss, val_acc = self._evaluate()

            if self.log_config is None:
                print("Epoch [{}]   Train: loss {:.4f} accuracy {:.4f}   Validation: loss {:.4f} accuracy {:.4f}".format(epoch, tr_loss, tr_acc, val_loss, val_acc))

