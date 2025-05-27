import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline

class TransformerEncoder(nn.Module):
    def __init__(self, device = 'cuda:0'):
        super(TransformerEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
        self.model = AutoModelWithLMHead.from_pretrained("DeepChem/ChemBERTa-77M-MLM").requires_grad_(False)
        # torch.save(self.model, "/mnt/disk1/xueying/jtvae/model_full_py_init2.pth")
        self.fc = nn.Linear(600, 420)
        self.fc2 = nn.Linear(600, 420)
        self.device = device
        # self.feature_extractor = feature_extractor()
        # self.feature_extractor = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1, return_tensors="pt")
    
    def forward(self, input_smiles):
        # print(input_smiles)
        features = self.feature_extractor(input_smiles) # torch.Size([8, 66, 600]): batch_size, sequence_length, hidden_size
        
        # 直接求mean可能不对，因为有padding
        def masked_mean_pooling(hidden_states, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)

        pooled_features = self.fc(torch.mean(features.logits, dim=1))

        # print(f"feature shape: {pooled_features.shape}")
        return pooled_features, self.fc2(features.logits)
    
    def feature_extractor(self, input_smiles):    
        model_inputs = self.tokenizer(input_smiles, return_tensors="pt", padding=True, truncation=True)
        for k, v in model_inputs.items():
            if type(v) is torch.Tensor:
                model_inputs[k] = v.to(self.device)
        # torch.save(self.model, "/mnt/disk1/xueying/jtvae/model_full_py.pth")
        # self.model = torch.load("/mnt/disk1/xueying/jtvae/model_full_ipynb.pth")
        features = self.model(**model_inputs)

        return features