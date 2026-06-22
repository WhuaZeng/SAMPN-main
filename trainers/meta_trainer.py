import torch
import torch.nn as nn
import numpy as np

class MetaTrainer:
    """元学习训练器"""
    
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_accuracies = []
    
    def train_episode(self, support_images, support_labels, query_images, query_labels):
        """训练一个元学习episode"""
        self.model.train()
        
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        self.optimizer.zero_grad()
        
        if isinstance(self.model, torch.nn.Module) and hasattr(self.model, 'predict'):
                        
            support_embeddings = self.model(support_images)
            query_embeddings = self.model(query_images)
            logits = self.model.predict(support_embeddings, support_labels, query_embeddings)
        else:
                    
            logits = self.model.meta_predict(support_images, support_labels, query_images,training = True)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, query_labels)
        
        loss.backward()
        self.optimizer.step()
        
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == query_labels).float().mean()
        
        return loss.item(), accuracy.item()
    
    def validate_episode(self, support_images, support_labels, query_images, query_labels):
        """验证一个episode"""
        self.model.eval()
        
        with torch.no_grad():
                      
            support_images = support_images.to(self.device)
            support_labels = support_labels.to(self.device)
            query_images = query_images.to(self.device)
            query_labels = query_labels.to(self.device)
            
            if isinstance(self.model, torch.nn.Module) and hasattr(self.model, 'predict'):
                support_embeddings = self.model(support_images)
                query_embeddings = self.model(query_images)
                logits = self.model.predict(support_embeddings, support_labels, query_embeddings)
            else:
                logits = self.model.meta_predict(support_images, support_labels, query_images,training = False)
            
            pred = torch.argmax(logits, dim=1)
            accuracy = (pred == query_labels).float().mean()
        
        return accuracy.item()
    
    def train_meta_epochs(self, train_sampler, val_sampler, epochs=100):
        """训练多个元学习epoch"""
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            num_episodes = 0
            
            for episode, episode_data in enumerate(train_sampler):
                                   
                if len(episode_data) == 5:
                    support_images, support_labels, query_images, query_labels, _ = episode_data
                else:
                    support_images, support_labels, query_images, query_labels = episode_data
                    
                loss, acc = self.train_episode(support_images, support_labels, query_images, query_labels)
                epoch_loss += loss
                epoch_acc += acc
                num_episodes += 1
                
                if episode % 10 == 0:
                    print(f'Epoch {epoch+1}, Episode {episode}: Loss={loss:.4f}, Acc={acc:.4f}')
            
            avg_loss = epoch_loss / num_episodes if num_episodes > 0 else 0
            avg_acc = epoch_acc / num_episodes if num_episodes > 0 else 0
            self.train_losses.append(avg_loss)
            
            val_acc = self.validate_meta_epoch(val_sampler)
            self.val_accuracies.append(val_acc)
            
            print(f'Epoch {len(self.train_losses)}: Avg Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, Val Acc={val_acc:.4f}')
    
    def validate_meta_epoch(self, val_sampler, num_episodes=20):
        """验证整个epoch"""
        total_acc = 0
        count = 0
        
        for episode, episode_data in enumerate(val_sampler):
            if episode >= num_episodes:
                break
            
            if len(episode_data) == 5:
                support_images, support_labels, query_images, query_labels, _ = episode_data
            else:
                support_images, support_labels, query_images, query_labels = episode_data
                
            acc = self.validate_episode(support_images, support_labels, query_images, query_labels)
            total_acc += acc
            count += 1
        
        return total_acc / count if count > 0 else 0

    def train_one_epoch(self, train_sampler, val_sampler):
        """训练单个epoch并返回指标"""
        epoch_loss = 0
        epoch_acc = 0
        num_episodes = 0
        
        for episode, episode_data in enumerate(train_sampler):
                               
            if len(episode_data) == 5:
                support_images, support_labels, query_images, query_labels, _ = episode_data
            else:
                support_images, support_labels, query_images, query_labels = episode_data
                
            loss, acc = self.train_episode(support_images, support_labels, query_images, query_labels)
            epoch_loss += loss
            epoch_acc += acc
            num_episodes += 1
            
            if episode % 10 == 0:
                print(f'Epoch {len(self.train_losses)+1}, Episode {episode}: Loss={loss:.4f}, Acc={acc:.4f}')
                
        avg_loss = epoch_loss / num_episodes if num_episodes > 0 else 0
        avg_acc = epoch_acc / num_episodes if num_episodes > 0 else 0
        self.train_losses.append(avg_loss)
                
        val_acc = self.validate_meta_epoch(val_sampler)
        self.val_accuracies.append(val_acc)
                
        print(f'Epoch {len(self.train_losses)}: Avg Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, Val Acc={val_acc:.4f}')