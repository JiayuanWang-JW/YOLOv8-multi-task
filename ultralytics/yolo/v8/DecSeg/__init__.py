# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import MultiPredictor, predict
from .train import DetectionSegmentationTrainer, train
from .val import MultiValidator, val

__all__ = 'MultiPredictor', 'predict', 'DetectionSegmentationTrainer', 'train', 'MultiValidator', 'val'
