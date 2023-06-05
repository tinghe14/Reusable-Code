import torch.nn as nn
from config import *
import torch
from transformers import AutoModel

class BioBERTTagger(nn.Module):
    def __init__(self):
        super(BioBERTTagger, self).__init__()  
        self.bert_base = AutoModel.from_pretrained(PRE_TRAIN_MODEL)
        self.linear = nn.Linear(768, 8) 
        #crf: input: number of tags
        weight = torch.cat(
            (torch.ones(8 - 1), torch.tensor([0.1]))
            ).to(torch.device("cuda" if torch.has_cuda else "cpu"))
        self.loss_function = nn.CrossEntropyLoss(weight=weight, ignore_index=0)

    def forward(self, input_ids, attention_mask, targets=None):  
        encoder_out = self.bert_base(input_ids, attention_mask).last_hidden_state
        logits = self.linear(encoder_out)
        # get rid of predicting '<PAD>' in argmax
        logits[:, :, 0] = float('-Inf') 
        pred = torch.argmax(logits, dim=-1)
        tag_seq = []
        for i, s in enumerate(input_ids):
            length = torch.sum(s > 1) # excluding special tokens 1
            length -= 2
            tag_seq.append(pred[i, 1:length + 1])
        output = {"tag_seq": tag_seq}
        if targets is not None:
            loss = self.loss_function(logits.transpose(-1, -2), targets)
            output["loss"] = loss
        return output
    
if __name__ == '__main__':
    model = BioBERTTagger()
