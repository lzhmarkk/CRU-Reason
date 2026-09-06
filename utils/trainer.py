import os
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Optional, Type, List
from tqdm.autonotebook import trange

from pyhealth.trainer import Trainer, logger, is_best
from utils.metrics import get_metrics_fn


class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print("Using device: ", device)
        kwargs['device'] = device

        super().__init__(**kwargs)

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              test_dataloader: Optional[DataLoader] = None,
              epochs: int = 5,
              patience: int = 5,
              optimizer_class: Type[Optimizer] = torch.optim.Adam,
              optimizer_params: Optional[Dict[str, object]] = None,
              steps_per_epoch: int = None,
              evaluation_steps: int = 1,
              weight_decay: float = 0.0,
              max_grad_norm: float = None,
              monitor: Optional[str] = None,
              monitor_criterion: str = "max",
              load_best_model_at_last: bool = True,
              resume_epoch: bool = -1
              ):
        if optimizer_params is None: optimizer_params = {"lr": 1e-3}

        # logging
        logger.info("Training:")
        logger.info(f"Batch size: {train_dataloader.batch_size}")
        logger.info(f"Optimizer: {optimizer_class}")
        logger.info(f"Optimizer params: {optimizer_params}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"Max grad norm: {max_grad_norm}")
        logger.info(f"Val dataloader: {val_dataloader}")
        logger.info(f"Monitor: {monitor}")
        logger.info(f"Monitor criterion: {monitor_criterion}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Patience: {patience}")

        # set optimizer
        param = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in param if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        # initialize
        data_iterator = iter(train_dataloader)
        best_score = -1 * float("inf") if monitor_criterion == "max" else float("inf")
        best_epoch = -1
        if steps_per_epoch == None: steps_per_epoch = len(train_dataloader)
        global_step = 0

        # epoch training loop
        for epoch in range(epochs):
            training_loss = []
            self.model.zero_grad()
            self.model.train()

            if epoch < resume_epoch:
                print(f"Skip epoch {epoch}, until epoch {resume_epoch}")
                continue
            elif epoch == resume_epoch:
                print(f"Resuming at epoch {resume_epoch}")
                self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

            # batch training loop
            logger.info("")
            for _ in trange(steps_per_epoch, desc=f"Epoch {epoch} / {epochs}", smoothing=0.05):
                try:
                    data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(train_dataloader)
                    data = next(data_iterator)

                # forward
                output = self.model(**data)
                loss = output["loss"]
                # backward
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                # update
                optimizer.step()
                optimizer.zero_grad()
                training_loss.append(loss.item())
                global_step += 1

            # log and save
            logger.info(f"--- Train epoch-{epoch}, step-{global_step} ---")
            logger.info(f"loss: {sum(training_loss) / len(training_loss):.4f}")

            # validation
            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)[0]
                logger.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
                for key in scores.keys():
                    logger.info("{}: {:.4f}".format(key, scores[key]))
                # save best model
                if monitor is not None:
                    score = scores[monitor]
                    if is_best(best_score, score, monitor_criterion):
                        logger.info(f"New best {monitor} score ({score:.4f}) at epoch-{epoch}, step-{global_step}")
                        best_score = score
                        best_epoch = epoch
                        if self.exp_path is not None:
                            self.save_ckpt(os.path.join(self.exp_path, "best.ckpt"))

                    elif epoch - best_epoch > patience:
                        logger.info(f"Early stopping at epoch-{epoch}, step-{global_step}")
                        break

        # load best model
        if load_best_model_at_last and self.exp_path is not None and os.path.isfile(
                os.path.join(self.exp_path, "best.ckpt")):
            logger.info("Loaded best model")
            self.load_ckpt(os.path.join(self.exp_path, "best.ckpt"))

    def evaluate(self, dataloader) -> List[Dict[str, float]]:
        assert self.model.mode is not None
        y_true_all, y_prob_all, loss_mean = self.inference(dataloader)

        mode = self.model.mode
        metrics_fn = get_metrics_fn(mode)
        scores_multi = []
        if mode == 'binary':
            for threshold in range(1, 10):
                scores = metrics_fn(y_true_all, y_prob_all, threshold=threshold / 10)
                scores["loss"] = loss_mean
                scores_multi.append(scores)
        else:
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                scores = metrics_fn(y_true_all, y_prob_all, threshold=threshold)
                scores["loss"] = loss_mean
                scores_multi.append(scores)
        return scores_multi
