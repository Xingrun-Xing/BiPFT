import torch
import logging
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
logging.basicConfig(level=logging.INFO)

class KDLearner(object):
    def __init__(self, student_model, teacher_model):
        self.student_model = student_model
        self.teacher_model = teacher_model.cuda()
        print('#'*50,next(self.teacher_model.parameters()).device)
        self.teacher_model.train()
        self.loss_mse = MSELoss()
        self.temperature = 1.

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()
    
    def masked_soft_cross_entropy(self, predicts, targets, real_label):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)

        res = (- targets_prob * student_likelihood).mean(-1)
        res = torch.where(real_label==-100, torch.zeros_like(res).cuda(), res)

        real_label_nums = torch.where(real_label==-100, torch.zeros_like(real_label).cuda(), torch.ones_like(real_label).cuda()).sum(dim=-1).unsqueeze(-1)
        
        return (res/real_label_nums).sum(-1).mean()
    
    def __call__(self, batch):
        student_outputs = self.student_model(**batch)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**batch)

        cls_dist_loss = self.soft_cross_entropy(student_outputs.seq_relationship_logits / self.temperature, teacher_outputs.seq_relationship_logits / self.temperature)
        
        rep_dist_loss_layerwise = []
        rep_dist_loss = 0.
        for student_rep, teacher_rep in zip(student_outputs.hidden_states, teacher_outputs.hidden_states):
            tmp_loss = self.loss_mse(student_rep, teacher_rep)
            rep_dist_loss += tmp_loss
            rep_dist_loss_layerwise.append(tmp_loss.item())
        rep_dist_loss = rep_dist_loss / len(rep_dist_loss_layerwise)
        
        real_loss = student_outputs.loss

        total_loss = cls_dist_loss + rep_dist_loss + real_loss
        return {'total_loss':total_loss, 'cls_dist_loss':cls_dist_loss, 'mlm_dist_loss':0, 'rep_dist_loss':rep_dist_loss, 'real_loss':real_loss}

