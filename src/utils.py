from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach, is_sagemaker_mp_enabled
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from sklearn.metrics import f1_score
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_only, smp_nested_concat

# use f1-macro as default metric for evaluation
def compute_metrics_da(eval_pred):
    logits, labels = eval_pred
    # labels[0] are class_labels and labels[1] are topic_labels
    # (according to label_names=['class_labels', 'topic_labels'] and detaching in trainer.prediction_step(...)
    class_labels = labels[0]
    topic_labels = labels[1]
    # logits[0] are class logits and logits[1] are topic logits
    class_predictions = np.argmax(logits[0], axis=-1)
    topic_predictions = np.argmax(logits[1], axis=-1)
    f1_bin = f1_score(class_labels, class_predictions, average='binary', pos_label=1) if len(set(class_labels))==2 else None
    result = {
        'f1_binary': f1_bin,
        'f1_macro': f1_score(y_true=class_labels, y_pred=class_predictions, average='macro'),
        'f1_micro': f1_score(y_true=class_labels, y_pred=class_predictions, average='micro'),
        'topic_f1_macro': f1_score(y_true=topic_labels, y_pred=topic_predictions, average='macro'),
        'topic_f1_micro': f1_score(y_true=topic_labels, y_pred=topic_predictions, average='micro')
    }

    return result

class DomainAdaptationTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss =  outputs[0]
        # class_loss_per_sample = outputs[1]
        # topic_loss_per_sample = outputs[2]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


def convert_timediffs_to_timebins(time_diffs):
    time_diffs = sorted(time_diffs)
    start = 1 # we start at 1 because we need an initial bin (without data) for the offset computation in model.forward
    conversion = {}
    for i in time_diffs:
        if i not in conversion.keys():
            conversion[i]  = start
            start += 1
    return conversion