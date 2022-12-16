from transformers import Trainer,AutoModelForSequenceClassification, EarlyStoppingCallback 
from transformers import AutoTokenizer
from transformers.models.roberta import RobertaPreTrainedModel
from transformers import TrainingArguments
from transformers import set_seed
from transformers.models.distilbert import DistilBertConfig
import torch
from torch import nn
from transformers.models.distilbert.modeling_distilbert import *

class AugmentedDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        
        
        self.total_num_features = config.dim + config.num_extra_features
        
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(self.total_num_features, self.total_num_features)
        self.classifier = nn.Linear(self.total_num_features, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        numeric_features: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
    
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        images_features = None
        
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        cls_embeds = hidden_state[:, 0]  # (bs, dim) THIS IS THE CLS EMBEDDING
        
        
        features = [cls_embeds,numeric_features,images_features]
        features = torch.cat([f for f in features if f is not None], dim=-1) #TODO: Include image features here
        
        logits = self.pre_classifier(features)  # (bs, dim)
        logits = nn.ReLU()(logits)  # (bs, dim)
        logits = self.dropout(logits)  # (bs, dim)
        logits = self.classifier(logits)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


def get_model(model_name,
              seed,
              num_numeric_features = 0,
              num_image_features = 0,
              problem_type = 'regression',
              num_labels = 1):
    
    set_seed(seed)
    config = DistilBertConfig.from_pretrained(model_name,
                                              problem_type = problem_type,
                                              num_labels = num_labels)

    config.num_extra_features = num_numeric_features +  num_image_features

    return AugmentedDistilBertForSequenceClassification(config)