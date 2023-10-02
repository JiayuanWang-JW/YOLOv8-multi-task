# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader, build_yolo_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr, ops
from ultralytics.yolo.utils.loss import BboxLoss, FocalLossV1, tversky
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.nn.tasks import MultiModel
from copy import copy
from ultralytics.yolo.utils.ops import crop_mask, xyxy2xywh, xywh2xyxy
import torch.nn.functional as F
import itertools
# BaseTrainer python usage
class DetectionSegmentationTrainer(BaseTrainer):
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a DetectionSegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'multi'
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        try:
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        except:
            gs = max(max(itertools.chain.from_iterable(de_parallel(self.model).stride)) if self.model else 0, 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        if self.args.v5loader:
            LOGGER.warning("WARNING ⚠️ 'v5loader' feature is deprecated and will be removed soon. You can train using "
                           'the default YOLOv8 dataloader instead, no argument is needed.')
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return create_dataloader(path=dataset_path,
                                     imgsz=self.args.imgsz,
                                     batch_size=batch_size,
                                     stride=gs,
                                     hyp=vars(self.args),
                                     augment=mode == 'train',
                                     cache=self.args.cache,
                                     pad=0 if mode == 'train' else 0.5,
                                     rect=self.args.rect or mode == 'val',
                                     rank=rank,
                                     workers=self.args.workers,
                                     close_mosaic=self.args.close_mosaic != 0,
                                     prefix=colorstr(f'{mode}: '),
                                     shuffle=mode == 'train',
                                     seed=self.args.seed)[0]
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        for count in range(len(batch)):
            batch[count]['img'] = batch[count]['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO multi model."""
        multi_model = MultiModel(cfg, nc=self.data['tnc'], verbose=verbose and RANK == -1)
        if weights:
            multi_model.load(weights)

        return multi_model

    def get_validator(self):
        """Return DetectionValidator and SegmentationValidator for validation of YOLO model."""
        self.loss_names = {"det": ['box_loss', 'cls_loss', 'dfl_loss'],"seg": ['Tv_loss', 'FL_loss']}
        return v8.DecSeg.MultiValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def criterion(self, preds, batch, name=None, count=None):
        """Compute loss for YOLO prediction and ground-truth."""
        if 'det' in name:
            self.compute_loss = Loss(de_parallel(self.model),count-len(self.data['labels_list']))
        elif 'seg' in name:
            self.compute_loss = SegLoss(de_parallel(self.model), overlap=self.args.overlap_mask, count=count-len(self.data['labels_list']), task_name = name, map=self.data['map'][count])
        return self.compute_loss(preds, batch)


    def label_loss_items(self, loss_items=None, prefix='train',task=None):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        if loss_items is not None:
            loss_names = []
            for name in self.data['labels_list']:
                loss_names.extend(self.loss_names[name[:3]])
            keys = [f'{prefix}/{x}' for x in loss_names]
            losses = [loss_withdraw for loss_withdraw in loss_items]
            loss_values = list(itertools.chain(*[l.tolist() for l in losses]))
            loss_items = [round(float(x), 5) for x in loss_values]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            keys = [f'{prefix}/{x}' for x in self.loss_names[task]]
            return keys

    def label_loss_items_val(self, loss_items=None, prefix='val', task=None):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        if loss_items is not None:
            keys = [f'{prefix}/{x}' for x in self.loss_names[task[:3]]]
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            keys = [f'{prefix}/{x}' for x in self.loss_names[task]]
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        ######Jiayuan
        loss_names = []
        for name in self.data['labels_list']:
            loss_names.extend(self.loss_names[name[:3]])
        return ('\n' + '%11s' *
                (4 + len(loss_names))) % ('Epoch', 'GPU_mem', *loss_names, 'Instances', 'Size')
        ######

    def plot_training_samples(self, batch, ni, task=None):
        ######Jiayuan
        """Plots training samples with their annotations."""
        fname = self.save_dir / f"train_batch{self.data['labels_list'][task]}{ni}.jpg" if task!=None else self.save_dir / f'train_batch{ni}.jpg'
        if 'det' in self.data['labels_list'][task]:
            plot_images(images=batch['img'],
                        batch_idx=batch['batch_idx'],
                        cls=batch['cls'].squeeze(-1),
                        bboxes=batch['bboxes'],
                        paths=batch['im_file'],
                        fname=fname,
                        on_plot=self.on_plot)
        elif 'seg' in self.data['labels_list'][task]:
            plot_images(images=batch['img'],
                        batch_idx=batch['batch_idx'],
                        cls=batch['cls'].squeeze(-1),
                        bboxes=batch['bboxes'],
                        masks=batch['masks'],
                        paths=batch['im_file'],
                        fname=fname,
                        on_plot=self.on_plot)
        ######

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        # plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png
        # plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        for i in range(len(self.train_loader.dataset.labels)):
            boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels[i]], 0)
            cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels[i]], 0)
            plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)



# Criterion class for computing training losses
class Loss:

    def __init__(self, model,count):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[count]  # Detect() module_
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)



# Criterion class for computing training losses
class SegLoss(Loss):

    def __init__(self, model, overlap=True, count = None, task_name=None,map=None):  # model must be de-paralleled
        super().__init__(model,count)
        self.overlap = overlap
        self.map = map
        self.focal_loss = FocalLossV1()
        self.TL = tversky()
        self.sigmoid = nn.Sigmoid()
        self.task_name = task_name

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        ###### Jiayuan: This code only for binary segmentation. To Do: add multi seg
        loss = torch.zeros(2, device=self.device)  # dice FL
        batch_size = len(batch['im_file'])
        masks = batch['masks'].to(self.device).float()
        # import matplotlib.pyplot as plt
        # gt_masks = masks.cpu().numpy()
        # plt.imshow(gt_masks[0], cmap='gray')
        # plt.title('imgPredict')
        # plt.axis('off')
        # plt.savefig('/home/jiayuan/ultralytics-main/ultralytics/runs/test_gt.png')
        gt_masks = masks.unsqueeze(1).clamp_(max=1)
        neg_mask = 1 - gt_masks
        binary_mask = torch.cat((neg_mask, gt_masks), dim=1)
        loss[0] = self.TL(preds, binary_mask, 0.7)
        loss[1] = self.focal_loss(preds, binary_mask)
        loss[0] *= self.hyp.TL    # TL gain
        loss[1] *= self.hyp.FL  # FL gain
        # loss[3] = (preds * 0).sum() ###### This is a auxiliary loss, do not move it. For PDD multi GPU issue.
        return loss.sum()* batch_size, loss.detach()




def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionSegmentationTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()

