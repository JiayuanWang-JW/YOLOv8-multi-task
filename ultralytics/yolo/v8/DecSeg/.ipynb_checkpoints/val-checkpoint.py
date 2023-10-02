# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing.pool import ThreadPool
import posixpath
from ultralytics.yolo.data import build_dataloader, build_yolo_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, colorstr, ops, NUM_THREADS
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou, SegmentMetrics, mask_iou
from ultralytics.yolo.utils.plotting import output_to_target, plot_images, Annotator, Colors
from ultralytics.yolo.utils.torch_utils import de_parallel
import torch.nn as nn
import math
import contextlib
class MultiValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'multi'
        self.is_coco = False
        self.class_map = None
        self.metrics = []
        try:
            for name in dataloader.dataset.data['labels_list']:
                if 'det' in name:
                    self.metrics.append(DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot))
                if 'seg' in name:
                    self.metrics.append(SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot))
        except:
            pass
        self.metrics_det = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.metrics_seg = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch_list = []
        for i,subbatch in enumerate(batch):
            if 'det' in self.data['labels_list'][i]:
                subbatch['img'] = subbatch['img'].to(self.device, non_blocking=True)
                subbatch['img'] = (subbatch['img'].half() if self.args.half else subbatch['img'].float()) / 255
                for k in ['batch_idx', 'cls', 'bboxes']:
                    subbatch[k] = subbatch[k].to(self.device)
                nb = len(subbatch['img'])
                self.lb = [torch.cat([subbatch['cls'], subbatch['bboxes']], dim=-1)[subbatch['batch_idx'] == i]
                           for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling
                batch_list.append(subbatch)
            elif 'seg' in self.data['labels_list'][i]:
                subbatch['img'] = subbatch['img'].to(self.device, non_blocking=True)
                subbatch['img'] = (subbatch['img'].half() if self.args.half else subbatch['img'].float()) / 255

                # for k in ['batch_idx', 'cls', 'bboxes']:
                #     subbatch[k] = subbatch[k].to(self.device)

                nb = len(subbatch['img'])
                # self.lb = [torch.cat([batch['cls'], subbatch['bboxes']], dim=-1)[subbatch['batch_idx'] == i]
                #            for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling
                subbatch['masks'] = subbatch['masks'].to(self.device).float()
                batch_list.append(subbatch)
        return batch_list

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')  # is COCO
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        for metrics in self.metrics:
            metrics.names = self.names
            metrics.plot = self.args.plots
        self.confusion_matrix={name: ConfusionMatrix(nc=self.data['nc_list'][count]) for count, name in enumerate(self.data['labels_list'])}
        self.seen = {name: 0 for name in self.data['labels_list']}
        self.jdict = []
        self.stats = {name: [] for name in self.data['labels_list']}
        self.nt_per_class = {name: [] for name in self.data['labels_list']}
        ###################################
        self.plot_masks = {name: [] for name in self.data['labels_list']}
        if self.args.save_json:
            check_requirements('pycocotools>=2.0.6')
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster
        self.sigmoid = nn.Sigmoid()
        self.combine = []

    def get_desc_det(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def get_desc_seg(self):
        """Return a formatted description of evaluation metrics."""
        return ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Mask(P',
                                         'R', 'mAP50', 'mAP50-95)')

    def postprocess_det(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det)
        return preds

    def postprocess_seg(self, preds,count):
        """Postprocesses YOLO predictions and returns output detections with proto."""
        preds = self.sigmoid(preds)
        _, preds = torch.max(preds, 1)
        return preds

    def replace_elements_in_column(self, tensor, target_list, column_index):
        target_list = torch.tensor(target_list, dtype=torch.float32, device=self.device)
        mask = torch.any(tensor[:, column_index].unsqueeze(1) == target_list, dim=1)

        replacement_tensor = tensor.clone()
        replacement_tensor[mask, column_index] = target_list[0]

        return replacement_tensor

    def update_metrics_det(self, preds, batch, task_name=None):
        """Metrics."""
        if self.args.combine_class:
            for si, pred in enumerate(preds):
                idx = batch['batch_idx'] == si
                cls = batch['cls'][idx]
                cls = self.replace_elements_in_column(cls,self.args.combine_class,0)
                bbox = batch['bboxes'][idx]
                nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
                shape = batch['ori_shape'][si]
                correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
                self.seen[task_name] += 1

                if npr == 0:
                    if nl:
                        self.stats[task_name].append(
                            (correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                        if self.args.plots:
                            self.confusion_matrix[task_name].process_batch(detections=None, labels=cls.squeeze(-1))
                    continue

                # Predictions
                if self.args.single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                predn = self.replace_elements_in_column(predn,self.args.combine_class,5)
                ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space pred

                # Evaluate
                if nl:
                    height, width = batch['img'].shape[2:]
                    tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                        (width, height, width, height), device=self.device)  # target boxes
                    ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                    ratio_pad=batch['ratio_pad'][si])  # native-space labels
                    labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                    correct_bboxes = self._process_batch_det(predn, labelsn)
                    # TODO: maybe remove these `self.` arguments as they already are member variable
                    if self.args.plots:
                        self.confusion_matrix[task_name].process_batch(predn, labelsn)
                self.stats[task_name].append(
                    (correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

                # Save
                if self.args.save_json:
                    self.pred_to_json(predn, batch['im_file'][si])
                if self.args.save_txt:
                    file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                    self.save_one_txt(predn, self.args.save_conf, shape, file)
        else:
            for si, pred in enumerate(preds):
                idx = batch['batch_idx'] == si
                cls = batch['cls'][idx]
                bbox = batch['bboxes'][idx]
                nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
                shape = batch['ori_shape'][si]
                correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
                self.seen[task_name] += 1

                if npr == 0:
                    if nl:
                        self.stats[task_name].append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                        if self.args.plots:
                            self.confusion_matrix[task_name].process_batch(detections=None, labels=cls.squeeze(-1))
                    continue

                # Predictions
                if self.args.single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space pred

                # Evaluate
                if nl:
                    height, width = batch['img'].shape[2:]
                    tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                        (width, height, width, height), device=self.device)  # target boxes
                    ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                    ratio_pad=batch['ratio_pad'][si])  # native-space labels
                    labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                    correct_bboxes = self._process_batch_det(predn, labelsn)
                    # TODO: maybe remove these `self.` arguments as they already are member variable
                    if self.args.plots:
                        self.confusion_matrix[task_name].process_batch(predn, labelsn)
                self.stats[task_name].append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

                # Save
                if self.args.save_json:
                    self.pred_to_json(predn, batch['im_file'][si])
                if self.args.save_txt:
                    file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                    self.save_one_txt(predn, self.args.save_conf, shape, file)

    def update_metrics_seg(self, preds, batch, task_name=None):
        """Metrics."""
        batch_size = len(batch['im_file'])
        mask_list = batch['masks'].to(self.device).float()
        for count in range(batch_size):
            gt_mask = mask_list[count].clamp_(max=1)
            pred_mask = preds[count].squeeze()
            # import matplotlib.pyplot as plt
            # gt_masks = pred_mask.cpu().numpy()
            # plt.imshow(gt_masks, cmap='gray')
            # plt.title('imgPredict')
            # plt.axis('off')
            # plt.savefig('/home/jiayuan/ultralytics-main/ultralytics/runs/test_gt.png')
            self.seg_metrics[task_name].reset()
            self.seg_metrics[task_name].addBatch(pred_mask.cpu(), gt_mask.cpu())
            self.seg_result[task_name]['pixacc'].update(self.seg_metrics[task_name].pixelAccuracy())
            self.seg_result[task_name]['subacc'].update(self.seg_metrics[task_name].lineAccuracy())
            self.seg_result[task_name]['IoU'].update(self.seg_metrics[task_name].IntersectionOverUnion())
            self.seg_result[task_name]['mIoU'].update(self.seg_metrics[task_name].meanIntersectionOverUnion())
            if self.args.plots and self.batch_i < 3:
                self.plot_masks[task_name].append(pred_mask.cpu())  # filter top 15 to plot

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        for i, labels_name in enumerate(self.data['labels_list']):

            self.metrics[i].speed = self.speed
            self.metrics[i].confusion_matrix = self.confusion_matrix[labels_name]

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        results_dict = []
        for i, labels_name in enumerate(self.data['labels_list']):
            try:
                stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats[labels_name])]  # to numpy
                if len(stats) and stats[0].any():
                    self.metrics[i].process(*stats)
                self.nt_per_class[labels_name] = np.bincount(stats[-1].astype(int), minlength=self.data['nc_list'][i])  # number of targets per class
                results_dict.append(self.metrics[i].results_dict)
            except:
                pass
        return results_dict

    def print_results(self):
        """Prints training/validation set metrics per class."""
        ######Jiayuan
        for count, label_name in enumerate(self.data['labels_list']):
            if 'seg' in label_name:
                pf = '%22s' + ('%11s' + '%11.3g') * len(self.seg_result[label_name])
                if self.args.verbose and self.training and self.nc > 1:
                    class_index = int([key for key, value in self.data['map'][count].items()][0])
                    key_values = [(key, value.avg) for key, value in self.seg_result[label_name].items()]
                    LOGGER.info(pf % (self.names[class_index], *sum(key_values, ())))
            else:
                LOGGER.info(('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95'))
                pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics[count].keys)  # print format
                LOGGER.info(pf % ('all', self.seen[label_name], self.nt_per_class[label_name].sum(), *self.metrics[count].mean_results()))
                if self.nt_per_class[label_name].sum() == 0:
                    LOGGER.warning(
                        f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

            # Print results per class
            if self.args.verbose and not self.training and self.nc > 1 and len(self.stats[label_name]):
                for i, c in enumerate(self.metrics[count].ap_class_index):
                    if self.data['map'][count]!='None':
                        for key, val in self.data['map'][count].items():
                            if val == str(c):
                                class_index = int(key)
                        key_values = [(key, value.avg) for key, value in self.seg_result[label_name].items()]
                        LOGGER.info(pf % (self.names[class_index], *sum(key_values, ())))
                    else:
                        if self.args.single_cls:
                            LOGGER.info(pf % (self.names[self.args.combine_class[0]], self.seen[label_name], self.nt_per_class[label_name][c],
                                              *self.metrics[count].class_result(i)))
                        else:
                            LOGGER.info(pf % (self.names[c], self.seen[label_name], self.nt_per_class[label_name][c], *self.metrics[count].class_result(i)))
            elif self.args.verbose and not self.training and self.nc > 1:
                class_index = int([key for key, value in self.data['map'][count].items()][0])
                key_values = [(key, value.avg) for key, value in self.seg_result[label_name].items()]
                LOGGER.info(pf % (self.names[class_index], *sum(key_values, ())))



            if self.args.plots:
                for normalize in True, False:
                    self.confusion_matrix[label_name].plot(save_dir=self.save_dir,
                                               names=self.names.values(),
                                               normalize=normalize,
                                               on_plot=self.on_plot)
        ######
    def _process_batch_det(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)

    def _process_batch_seg(self, detections, labels, pred_masks=None, gt_masks=None, overlap=False, masks=False):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        if masks:
            if overlap:
                nl = len(labels)
                index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
                gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
                gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode='bilinear', align_corners=False)[0]
                gt_masks = gt_masks.gt_(0.5)
            iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
            ######Jiayuan

        else:  # boxes
            iou = box_iou(labels[:, 1:], detections[:, :4])

        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)

    def get_dataloader(self, dataset_path, batch_size):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        if self.args.v5loader:
            LOGGER.warning("WARNING ⚠️ 'v5loader' feature is deprecated and will be removed soon. You can train using "
                           'the default YOLOv8 dataloader instead, no argument is needed.')
            gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
            return create_dataloader(path=dataset_path,
                                     imgsz=self.args.imgsz,
                                     batch_size=batch_size,
                                     stride=gs,
                                     hyp=vars(self.args),
                                     cache=False,
                                     pad=0.5,
                                     rect=self.args.rect,
                                     workers=self.args.workers,
                                     prefix=colorstr(f'{self.args.mode}: '),
                                     shuffle=False,
                                     seed=self.args.seed)[0]

        dataset = self.build_dataset(dataset_path, batch=batch_size, mode='val')
        dataloader = build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)
        return dataloader



    def plot_val_samples(self, batch, ni, task=None):
        """Plots validation samples with bounding box labels."""
        fname = self.save_dir / f"val_batch{task}{ni}.jpg" if task != None else self.save_dir / f'val_batch{ni}.jpg'
        if 'det' in task:
            plot_images(batch['img'],
                        batch['batch_idx'],
                        batch['cls'].squeeze(-1),
                        batch['bboxes'],
                        paths=batch['im_file'],
                        fname=fname,
                        names=self.names,
                        on_plot=self.on_plot)
        elif 'seg' in task:
            plot_images(batch['img'],
                        batch['batch_idx'],
                        batch['cls'].squeeze(-1),
                        batch['bboxes'],
                        batch['masks'],
                        paths=batch['im_file'],
                        fname=fname,
                        names=self.names,
                        on_plot=self.on_plot)



    def plot_predictions(self, batch, preds, ni, task=None):
        """Plots batch predictions with masks and bounding boxes."""
        fname = self.save_dir / f"val_batch{task}{ni}_pred.jpg" if task != None else self.save_dir / f'val_batch{ni}_pred.jpg'
        if 'det' in task:
            plot_images(batch['img'],
                        *output_to_target(preds, max_det=15),
                        paths=batch['im_file'],
                        fname=fname,
                        names=self.names,
                        on_plot=self.on_plot)  # pred
        # elif 'seg' in task:
        #     self.show_seg_result_batch(batch['img'],
        #                 self.plot_masks[task],
        #                 fname)  # pred
        #     self.plot_masks[task].clear()
        elif 'seg' in task:
            self.plot_images_seg(batch['img'],
                            self.plot_masks[task],
                            batch['im_file'],
                            fname,
                            self.on_plot)  # pred
            self.plot_masks[task].clear()

    def plot_images_seg(self,
                        images,
                        masks=np.zeros(0, dtype=np.uint8),
                        paths=None,
                        fname='images.jpg',
                        on_plot=None):
        # Plot image grid with labels
        colors = Colors()
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()

        max_size = 1920  # max image size
        max_subplots = 16  # max image subplots, i.e. 4x4
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)

        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y:y + h, x:x + w, :] = im

        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

        # Annotate
        fs = int((h + w) * ns * 0.01)  # font size
        annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=None)
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
            if paths:
                annotator.text((x + 5, y + 5), text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
                # Plot masks

                image_masks = masks[i]
                image_masks = image_masks.cpu().numpy().astype(int)
                im = np.asarray(annotator.im).copy()
                mh, mw = image_masks.shape
                color = colors(0)
                if mh != h or mw != w:
                    mask = image_masks.astype(np.uint8)
                    mask = cv2.resize(mask, (w, h))
                    mask = mask.astype(bool)
                else:
                    mask = image_masks.astype(bool)
                with contextlib.suppress(Exception):
                    im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + np.array(color) * 0.6
                annotator.fromarray(im)
        annotator.im.save(fname)  # save
        if on_plot:
            on_plot(fname)


    def show_seg_result_batch(self, images, results, save_dir=None, palette=None):
        images = images.cpu().float().numpy()*255
        if palette is None:
            palette = np.random.randint(0, 255, size=(3, 3))
        palette[0] = [0, 0, 0]
        palette[1] = [0, 255, 0]
        palette[2] = [255, 0, 0]
        palette = np.array(palette)
        bs, _, h, w = images.shape
        assert palette.shape[0] == 3
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        batch_size = images.shape[0]
        output_images = []
        kernel = np.ones((5, 5), np.uint8)
        for idx in range(batch_size):
            img = images[idx].copy()
            img = img.transpose(1, 2, 0)
            result = results[idx]

            color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(palette):
                color_seg[result == label, :] = color

            color_seg = color_seg[..., ::-1]
            # color_mask = np.mean(color_seg, 2)
            # img_copy = img.copy()  # Create a copy of the image
            # img_copy[color_mask != 0] = img_copy[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
            # img_copy = img_copy.astype(np.uint8)
            # img_copy = cv2.resize(img_copy, (w, h), interpolation=cv2.INTER_LINEAR)
            #
            # output_images.append(img_copy)
            color_mask = np.mean(color_seg, 2)
            smoothed_mask = cv2.GaussianBlur(color_mask, (5, 5), 0)

            # Apply erosion operation to the smoothed mask
            eroded_mask = cv2.erode(smoothed_mask, kernel, iterations=1)

            img_copy = img.copy()
            img_copy[eroded_mask != 0] = img_copy[eroded_mask != 0] * 0.5 + color_seg[eroded_mask != 0] * 0.5
            img_copy = img_copy.astype(np.uint8)
            img_copy = cv2.resize(img_copy, (w, h), interpolation=cv2.INTER_LINEAR)

            output_images.append(img_copy)

        # Create a canvas to hold all images
        max_images_per_row = 4  # Adjust this value as needed
        num_rows = (batch_size + max_images_per_row - 1) // max_images_per_row
        canvas_h = num_rows * h
        canvas_w = max_images_per_row * w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for i, img in enumerate(output_images):
            row_idx = i // max_images_per_row
            col_idx = i % max_images_per_row
            canvas[row_idx * h:(row_idx + 1) * h, col_idx * w:(col_idx + 1) * w, :] = img

        if save_dir:
            save_path = posixpath.abspath(save_dir)
            cv2.imwrite(save_path, canvas)

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def pred_to_json_det(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)})

    def pred_to_json_seg(self, predn, filename, pred_masks):
        """Save one JSON result."""
        # Example result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pred_masks = np.transpose(pred_masks, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5),
                'segmentation': rles[i]})

    def eval_json_det(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats


    def eval_json_seg(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm')]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.keys[idx + 1]], stats[
                        self.metrics.keys[idx]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats


def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate trained YOLO model on validation dataset."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = MultiValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
