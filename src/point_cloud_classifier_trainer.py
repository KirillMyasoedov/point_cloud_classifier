import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from .criterions.my_loss import SpatialEmbLoss
from .stones_dataset import StonesDataset
from .models import get_model
from .utils.utils import AverageMeter, Cluster, Logger, Visualizer
from .utils import transforms as my_transforms

torch.backends.cudnn.benchmark = True


class PointCloudClassifierTrainer(object):
    def __init__(self, train_settings, val_settings, model_settings):
        self.train_settings = train_settings
        self.val_settings = val_settings

        # set device
        device = torch.device("cuda:0" if self.train_settings['cuda'] else "cpu")

        # set model
        self.model = get_model(model_settings['name'], model_settings['kwargs'])
        self.model.init_output(self.train_settings['loss_opts']['n_sigma'])
        self.model = torch.nn.DataParallel(self.model).to(device)

        # set criterion
        self.criterion = SpatialEmbLoss(**self.train_settings['loss_opts'])
        self.criterion = torch.nn.DataParallel(self.criterion).to(device)

        # set optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_settings['lr'], weight_decay=1e-4)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lambda_, )

        # clustering
        self.cluster = Cluster()

        # visualizer
        self.visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

        # logger
        self.logger = Logger(('train', 'val', 'iou'), 'loss')

        train_transform = my_transforms.get_transform([
            {
                'name': 'RandomCrop',
                'opts': {
                    'keys': ('image', 'instance', 'label'),
                    'size': (self.train_settings["image_size"], self.train_settings["image_size"]),
                },
            },
            {
                'name': 'ToTensor',
                'opts': {
                    'keys': ('image', 'instance', 'label'),
                    'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                }
            }
        ])
        val_transform = my_transforms.get_transform([
            {
                'name': 'ToTensor',
                'opts': {
                    'keys': ('image', 'instance', 'label'),
                    'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                }
            },
        ])

        # train dataloader
        train_dataset = StonesDataset(root_dir=self.train_settings["root_dir"],
                                      type=self.train_settings["type"],
                                      class_id=26,
                                      size=self.train_settings["size"],
                                      transform=train_transform)

        self.train_dataset_it = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.train_settings['batch_size'],
                                                            shuffle=True,
                                                            drop_last=True,
                                                            num_workers=self.train_settings['workers'],
                                                            pin_memory=True if self.train_settings['cuda'] else False)

        # val dataloader
        val_dataset = StonesDataset(root_dir=self.val_settings["root_dir"],
                                    type=self.val_settings["type"],
                                    class_id=26,
                                    transform=val_transform)

        self.val_dataset_it = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.val_settings['batch_size'],
                                                          shuffle=True,
                                                          drop_last=True,
                                                          num_workers=self.val_settings['workers'],
                                                          pin_memory=True if self.train_settings['cuda'] else False)

    def lambda_(self, epoch):
        return pow((1 - (epoch / self.train_settings['n_epochs'])), 0.9)

    def train(self, epoch):
        # define meters
        loss_meter = AverageMeter()

        # put model into training mode
        self.model.train()

        for param_group in self.optimizer.param_groups:
            print('learning rate: {}'.format(param_group['lr']))

        for i, sample in enumerate(tqdm(self.train_dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze()
            class_labels = sample['label'].squeeze()

            output = self.model(im)
            loss = self.criterion(output, instances, class_labels, **self.train_settings['loss_w'])
            loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.train_settings['display'] and i % self.train_settings['display_it'] == 0:
                with torch.no_grad():
                    self.visualizer.display(im[0], 'image')

                    predictions = self.cluster.cluster_with_gt(output[0], instances[0],
                                                               n_sigma=self.train_settings['loss_opts']['n_sigma'])
                    self.visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred')

                    sigma = output[0][2].cpu()
                    sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
                    sigma[instances[0] == 0] = 0
                    self.visualizer.display(sigma, 'sigma')

                    seed = torch.sigmoid(output[0][3]).cpu()
                    self.visualizer.display(seed, 'seed')

            loss_meter.update(loss.item())

        return loss_meter.avg

    def val(self, epoch):

        # define meters
        loss_meter, iou_meter = AverageMeter(), AverageMeter()

        # put model into eval mode
        self.model.eval()

        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_dataset_it)):

                im = sample['image']
                instances = sample['instance'].squeeze()
                class_labels = sample['label'].squeeze()

                output = self.model(im)
                loss = self.criterion(output, instances, class_labels, **self.train_settings['loss_w'], iou=True,
                                      iou_meter=iou_meter)
                loss = loss.mean()

                if self.train_settings['display'] and i % self.train_settings['display_it'] == 0:
                    with torch.no_grad():
                        self.visualizer.display(im[0], 'image')

                        predictions = self.cluster.cluster_with_gt(output[0], instances[0],
                                                                   n_sigma=self.train_settings['loss_opts']['n_sigma'])
                        self.visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred')

                        sigma = output[0][2].cpu()
                        sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
                        sigma[instances[0] == 0] = 0
                        self.visualizer.display(sigma, 'sigma')

                        seed = torch.sigmoid(output[0][3]).cpu()
                        self.visualizer.display(seed, 'seed')

                loss_meter.update(loss.item())

        return loss_meter.avg, iou_meter.avg

    def save_checkpoint(self, state, is_best, name='checkpoint.pth'):
        print('=> saving checkpoint')
        file_name = os.path.join(self.train_settings['save_dir'], name)
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(
                self.train_settings['save_dir'], 'best_iou_model.pth'))

    def start_training(self):
        # resume
        start_epoch = 0
        best_iou = 0
        if self.train_settings['resume_path'] is not None and os.path.exists(self.train_settings['resume_path']):
            print('Resuming model from {}'.format(self.train_settings['resume_path']))
            state = torch.load(self.train_settings['resume_path'])
            start_epoch = state['epoch'] + 1
            best_iou = state['best_iou']
            self.model.load_state_dict(state['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(state['optim_state_dict'])
            self.logger.data = state['logger_data']

        for epoch in range(start_epoch, self.train_settings['n_epochs']):

            print('Starting epoch {}'.format(epoch))
            # self.scheduler.step(epoch)

            train_loss = self.train(epoch)
            val_loss, val_iou = self.val(epoch)

            print('===> train loss: {:.2f}'.format(train_loss))
            print('===> val loss: {:.2f}, val iou: {:.2f}'.format(val_loss, val_iou))

            self.logger.add('train', train_loss)
            self.logger.add('val', val_loss)
            self.logger.add('iou', val_iou)
            self.logger.plot(save=self.train_settings['save'], save_dir=self.train_settings['save_dir'])

            is_best = val_iou > best_iou
            best_iou = max(val_iou, best_iou)

            if self.train_settings['save']:
                state = {
                    'epoch': epoch,
                    'best_iou': best_iou,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optimizer.state_dict(),
                    'logger_data': self.logger.data
                }
                self.save_checkpoint(state, is_best)
