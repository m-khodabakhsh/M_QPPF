import torch
import torch.nn as nn
from transformers import BertModel

class M_QPPF(nn.Module):
    def __init__(self, config):
        super(M_QPPF, self).__init__()

        self.DOCperQuery = config['Doc_per_query']
        self.bert = BertModel.from_pretrained(config['bertModel'])

        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, config['num_labels'])
        self.classifier.apply(self._init_weights)
        #LSTM
        self.RNN = torch.nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size,
                                  num_layers=1, bias=True, batch_first=True, dropout=0.2)
        #GRU
        # self.RNN = torch.nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size,
        #                          num_layers=1, bias=True, batch_first=True, dropout=0.2)

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 100),
            torch.nn.Linear(100, config['num_labels']),
            torch.nn.Sigmoid()
        )
        self.regression.apply(self._init_weights)

    def forward(self,input_ids: torch.Tensor,attention_mask: [torch.Tensor],token_type_ids: [torch.Tensor]) :
        model_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        dropout = self.dropout((model_outputs.last_hidden_state)[:, 0])
        rank_logits = self.classifier(dropout)
        rank_logits = rank_logits.view(-1, self.DOCperQuery)

        res = dropout.view(-1, self.DOCperQuery,self.bert.config.hidden_size)
        lstm_output, recent_hidden = self.RNN(res)
        recent_hidden = recent_hidden[0].squeeze(0)
        qpp_logits =  self.regression(recent_hidden)

        return rank_logits , qpp_logits


    def _init_weights(self,module: nn.Module) -> None:  # type: ignore
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()