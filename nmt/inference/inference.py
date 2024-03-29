import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

class Inference:
    def __init__(self,
                 dataloader,
                 model,
                 criterion,
                 metric_fns,
                 optimizer,
                 device,
                 valid_dataloader = None,
                 lr_scheduler = None,
                 num_epochs = None):

        self.dataloader = dataloader
        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None
        
        self.model = model

        self.criterion = criterion
        self.metric_fns = metric_fns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.start_epoch = 1
        
        self.device = device

    def model_switch(self, mode = "training"):
        if mode == "training":
            self.model.train()
        else:
            self.model.eval()


    def _train_epoch(self, epoch):
        self.model_switch(mode = "training")

        curr_metrics = [0 for met in self.metric_fns]
        curr_loss = 0
        num_batches = 0

        for batch_idx, (data, inp_target, out_target) in enumerate(tqdm(self.dataloader)):
            num_batches += 1
            data, inp_target, out_target = data.to(self.device), inp_target.to(self.device), out_target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data, inp_target)
            loss = self.criterion(output, F.one_hot(out_target, num_classes = self.dataloader.tokenizer.vocab_size + 4))
            loss.backward()
            self.optimizer.step()

            curr_loss += loss.item()

            # train_metrics = {}
            for i, met in enumerate(self.metric_fns):
                curr_metrics[i] += met(output, out_target)
                # train_metrics["training_" + met.__name__] = met(output, out_target)
           
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        train_metrics = {"loss" : curr_loss / num_batches}
        for i, met in enumerate(self.metric_fns):
            train_metrics[met.__name__] = curr_metrics[i] / num_batches
            
        return train_metrics
        
    def _valid_epoch(self, epoch):
        self.model_switch(mode = "inference")

        curr_metrics = [0 for met in self.metric_fns]
        curr_loss = 0
        num_batches = 0


        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.valid_dataloader)):
                num_batches += 1

                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data, target)
                loss = self.criterion(output, target)

                curr_loss += loss.item()

                for i, met in enumerate(self.metric_fns):
                    curr_metrics[i] += met(output, target)

            val_metrics = {"loss" : curr_loss / num_batches}
            for i, met in enumerate(self.metric_fns):
                val_metrics[met.__name__] = curr_metrics[i] / num_batches
            
            return val_metrics

    def train(self):
        for epoch in trange(self.start_epoch, self.num_epochs + 1):
            train_metrics = self._train_epoch(epoch)

            val_metrics = None
            if self.do_validation:
                val_metrics = self._valid_epoch(epoch)
            
            print(train_metrics, val_metrics)

























# class Trainer:
#     """
#     Trainer class
#     """
#     def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
#                  data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
#         super().__init__(model, criterion, metric_ftns, optimizer, config)
        
#         self.config = config
#         self.device = device
#         self.data_loader = data_loader
#         if len_epoch is None:
#             # epoch-based training
#             self.len_epoch = len(self.data_loader)
#         else:
#             # iteration-based training
#             self.data_loader = inf_loop(data_loader)
#             self.len_epoch = len_epoch
#         self.valid_data_loader = valid_data_loader
#         self.do_validation = self.valid_data_loader is not None
#         self.lr_scheduler = lr_scheduler
#         self.log_step = int(np.sqrt(data_loader.batch_size))

#         self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
#         self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

#     def _train_epoch(self, epoch):
#         """
#         Training logic for an epoch

#         :param epoch: Integer, current training epoch.
#         :return: A log that contains average loss and metric in this epoch.
#         """
#         self.model.train()
#         self.train_metrics.reset()
#         for batch_idx, (data, target) in enumerate(self.data_loader):
#             data, target = data.to(self.device), target.to(self.device)

#             self.optimizer.zero_grad()
#             output = self.model(data)
#             loss = self.criterion(output, target)
#             loss.backward()
#             self.optimizer.step()

#             self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
#             self.train_metrics.update('loss', loss.item())
#             for met in self.metric_ftns:
#                 self.train_metrics.update(met.__name__, met(output, target))

#             if batch_idx % self.log_step == 0:
#                 self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
#                     epoch,
#                     self._progress(batch_idx),
#                     loss.item()))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#             if batch_idx == self.len_epoch:
#                 break
#         log = self.train_metrics.result()

#         if self.do_validation:
#             val_log = self._valid_epoch(epoch)
#             log.update(**{'val_'+k : v for k, v in val_log.items()})

#         if self.lr_scheduler is not None:
#             self.lr_scheduler.step()
#         return log

#     def _valid_epoch(self, epoch):
#         """
#         Validate after training an epoch

#         :param epoch: Integer, current training epoch.
#         :return: A log that contains information about validation
#         """
#         self.model.eval()
#         self.valid_metrics.reset()
#         with torch.no_grad():
#             for batch_idx, (data, target) in enumerate(self.valid_data_loader):
#                 data, target = data.to(self.device), target.to(self.device)

#                 output = self.model(data)
#                 loss = self.criterion(output, target)

#                 self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
#                 self.valid_metrics.update('loss', loss.item())
#                 for met in self.metric_ftns:
#                     self.valid_metrics.update(met.__name__, met(output, target))
#                 self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

#         # add histogram of model parameters to the tensorboard
#         for name, p in self.model.named_parameters():
#             self.writer.add_histogram(name, p, bins='auto')
#         return self.valid_metrics.result()

#     def _progress(self, batch_idx):
#         base = '[{}/{} ({:.0f}%)]'
#         if hasattr(self.data_loader, 'n_samples'):
#             current = batch_idx * self.data_loader.batch_size
#             total = self.data_loader.n_samples
#         else:
#             current = batch_idx
#             total = self.len_epoch
#         return base.format(current, total, 100.0 * current / total)
