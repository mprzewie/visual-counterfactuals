import argparse
import os

import torchvision.transforms
from torch.utils.data import Subset

from PIL import  Image
# import model.auxiliary_model as auxiliary_model
import numpy as np
import torch
import yaml

# from explainer.counterfactuals import compute_counterfactual
# from explainer.eval import compute_eval_metrics
# from explainer.utils import get_query_distractor_pairs, process_dataset
from tqdm import tqdm
from utils.common_config import (
    get_imagenet_test_transform,
    get_model,
    get_test_dataloader,
    get_test_dataset,
    get_test_transform, get_test_transform_wo_normalize, normalize_cub, get_test_dataset_framed, get_test_transform_resize_wo_normalize, get_test_transform_resize
)
from utils.path import Path
from pathlib import Path as PathlibPath
import matplotlib.pyplot as plt
import torchvision.transforms.functional as ttf
