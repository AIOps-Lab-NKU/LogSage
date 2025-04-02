import os
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

class GraphTrainer:
    def __init__(self, model, graph, labels, train_mask, val_mask, test_mask, device='cpu'):
        self.model = model.to(device)
        self.graph = graph.to(device)
        self.labels = labels.to(device)
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        self.best_val_acc = 0
        self.best_model = None

    def train(self, epochs=100):
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.graph.x, self.graph.edge_index)
            loss = F.nll_loss(out[self.train_mask], self.labels[self.train_mask])
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                train_acc, _, _, _ = self.evaluate(self.train_mask)
                val_acc, val_prec, val_rec, val_f1 = self.evaluate(self.val_mask)
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model = self.model.state_dict()
                    print(f'New best validation accuracy: {val_acc:.4f}')

                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                      f'Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}, Val F1: {val_f1:.4f}')

        # Load best model for testing
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

    def evaluate(self, mask):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.graph.x, self.graph.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[mask] == self.labels[mask]).sum()
            acc = correct.item() / mask.sum().item()
            
            y_true = self.labels[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            return acc, prec, rec, f1

    def test(self):
        test_acc, test_prec, test_rec, test_f1 = self.evaluate(self.test_mask)
        print(f'Test Acc: {test_acc:.4f}, Test Precision: {test_prec:.4f}, '
              f'Test Recall: {test_rec:.4f}, Test F1: {test_f1:.4f}')
        
        y_true = self.labels[self.test_mask].cpu().numpy()
        y_pred = self.model(self.graph.x, self.graph.edge_index)[self.test_mask].argmax(dim=1).cpu().numpy()
        print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))