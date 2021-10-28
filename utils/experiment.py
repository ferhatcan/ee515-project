import os
from itertools import cycle

import torch
import torch.nn as nn
import numpy as np


class Experiment:
    def __init__(self, model: nn.Module, optimizer, criterion, params):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.params = params

    def adjust_lr(self, p):
        lr_0 = 0.01
        alpha = 10
        beta = 0.75
        lr = lr_0 / ((1 + alpha * p) ** beta)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_one_epoch(self, target_dataloader, source_dataloader, epoch_number):
        self.model.train()
        self.model.to(self.params.device)

        if self.params.only_source:
            len_dataloader = len(source_dataloader)
        else:
            len_dataloader = max(len(source_dataloader), len(target_dataloader))
        if len(source_dataloader) >= len(target_dataloader):
            zip_datasets = zip(source_dataloader, cycle(target_dataloader))
        else:
            zip_datasets = zip(cycle(source_dataloader), target_dataloader)

        for step, ((images_src, class_src), (images_tgt, _)) in enumerate(zip_datasets):
            p = float(step + epoch_number * len_dataloader) / (self.params.epoch_num * len_dataloader)
            weight_lambda = 2. / (1. + np.exp(-10 * p)) - 1

            # self.adjust_lr(p)

            if step > len_dataloader:
                break

            # prepare domain label
            size_src = len(images_src)
            size_tgt = len(images_tgt)
            label_src = torch.zeros(size_src).long().to(self.params.device)  # source 0
            label_tgt = torch.ones(size_tgt).long().to(self.params.device)  # target 1

            # make images variable
            class_src = class_src.to(self.params.device)
            images_src = images_src.to(self.params.device)

            # zero gradients for optimizer
            self.optimizer.zero_grad()

            # train on source domain
            src_class_output, src_domain_output = self.model(input_data=images_src, lambda_weight=weight_lambda)
            src_loss_class = self.criterion(src_class_output, class_src)
            src_loss_domain = self.criterion(src_domain_output, label_src)

            # train on target domain
            if not self.params.only_source:
                images_tgt = images_tgt.to(self.params.device)
                _, tgt_domain_output = self.model(input_data=images_tgt, lambda_weight=weight_lambda)
                tgt_loss_domain = self.criterion(tgt_domain_output, label_tgt)

                loss = src_loss_class + src_loss_domain + tgt_loss_domain
            else:
                tgt_loss_domain = src_loss_domain
                loss = src_loss_class

            # optimize
            loss.backward()
            self.optimizer.step()

            # if self.params.src_only_flag:
            #     loss = src_loss_class

            #@todo: add logger

            if (step + 1) % int(len_dataloader * self.params.log_step) == 0:
                print("Step [{:2d}/{}]: src_loss_class={:.6f}, "
                      "src_loss_domain={:.6f}, tgt_loss_domain={:.6f}, loss={:.6f}".
                      format(step + 1, len_dataloader, src_loss_class.data.item(),
                             src_loss_domain.data.item(), tgt_loss_domain.data.item(), loss.data.item()))

    def train(self, target_dataloader, source_dataloader, save_name=""):
        for epoch in range(self.params.epoch_num):
            print("Epoch [{:2d}/{}] ".format(epoch+1, self.params.epoch_num))

            self.train_one_epoch(target_dataloader['train'], source_dataloader['train'], epoch)

            if save_name == "":
                save_name = self.params.experiment_name + '_{:d}'.format(self.params.epoch_num)
            save_path = './trained_params/' + save_name + '_model_last.pth'
            self.save(save_path)

            print('\n---------------------------------------------------------')
            print('Source Domain Results: ')
            self.test(source_dataloader['validation'], domain_flag='source')
            print('Target Domain results: ')
            self.test(target_dataloader['validation'], domain_flag='target')
            print('---------------------------------------------------------\n')

    def test(self, dataloader, domain_flag='target'):
        self.model.eval()
        self.model.to(self.params.device)

        # init loss and accuracy
        loss_ = 0.0
        acc_ = 0.0
        acc_domain_ = 0.0
        n_total = 0

        # evaluate network
        for (images, labels) in dataloader:
            images = images.to(self.params.device)
            labels = labels.to(self.params.device)  # labels = labels.squeeze(1)
            size = len(labels)
            if domain_flag == 'target':
                labels_domain = torch.ones(size).long().to(self.params.device)
            else:
                labels_domain = torch.zeros(size).long().to(self.params.device)

            preds, domain = self.model(images, lambda_weight=0)

            loss_ += self.criterion(preds, labels).item()

            pred_cls = preds.data.max(1)[1]
            pred_domain = domain.data.max(1)[1]
            acc_ += pred_cls.eq(labels.data).sum().item()
            acc_domain_ += pred_domain.eq(labels_domain.data).sum().item()
            n_total += size

        loss = loss_ / n_total
        acc = acc_ / n_total
        acc_domain = acc_domain_ / n_total

        print("Avg Loss = {:.6f}, Avg Accuracy = {:.2%}, {}/{}, Avg Domain Accuracy = {:2%}".format(loss, acc, acc_,
                                                                                                    n_total,
                                                                                                    acc_domain))

        return loss, acc, acc_domain

    def save(self, save_path):
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print("save pretrained model to: {}".format(save_path))

    def load(self, load_path):
        if not os.path.exists(load_path):
            print('Model parameters cannot be found in {}!!!'.format(load_path))
        else:
            self.model.load_state_dict(torch.load(load_path))
            print('Model parameters are successfully loaded from {}'.format(load_path))

