# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchmetrics

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from protopool import PrototypeChooser


def get_query_distractor_pairs(
    dataset,
    confusion_matrix,
    max_num_distractors,
    seed=0,
):
    """
    Compute query-distractor image pairs using a confusion matrix:
        (1) For each query class, we select the most confusing
        class as it's distractor class.
        (2) Next, we randomly sample `max_num_distractors`
        from the distractor class as the distractor images.

    We return the result as a a dictionary.
    """
    np.random.seed(seed)
    result = {}

    # process all images
    for query_index in tqdm(list(range(len(dataset))), "get_query_distractor_pairs"):
        # determine the distractor class for this sample
        query_class = dataset.__getitem__(query_index)["target"]
        row = np.copy(confusion_matrix[query_class])
        row[query_class] = -1
        if np.all(row <= 0):  # no signal from confusion matrix - skip
            continue
        distractor_class = np.argmax(row)

        # gater all images belonging to distractor class
        distractor_index = dataset.get_target(distractor_class)

        # randomly select `max_num_distractors`
        num_random = min(len(distractor_index), max_num_distractors)
        distractor_index = np.random.choice(distractor_index, num_random, replace=False)
        distractor_index = distractor_index.reshape(-1).tolist()

        # add to dictionary
        result[query_index] = {
            "distractor_index": distractor_index,
            "distractor_class": distractor_class,
            "query_index": query_index,
            "query_class": query_class,
        }

        # if query_index > 10:
        #     break

    return result


def get_query_distractor_pairs_with_predetermined_targets(
    query_dataset,
    targets,
    ground_truth_dataset,
    max_num_distractors,
    seed=0,
):
    """
    Compute query-distractor image pairs using a confusion matrix:
        (1) For each query class, we select the most confusing
        class as it's distractor class.
        (2) Next, we randomly sample `max_num_distractors`
        from the distractor class as the distractor images.

    We return the result as a a dictionary.
    """
    np.random.seed(seed)
    result = {}

    # process all images
    for query_index in tqdm(list(range(len(query_dataset))), "get_query_distractor_pairs"):
        # determine the distractor class for this sample
        query_class = query_dataset.__getitem__(query_index)[1]
        
        distractor_class = targets[query_index]
        
        # TODO: I need to gather distractors from the *real* data...

        # gater all images belonging to distractor class
        distractor_index = ground_truth_dataset.get_target(distractor_class)
        # print(f"Found {len(distractor_index)} of class {distractor_class}")

        # randomly select `max_num_distractors`
        num_random = min(len(distractor_index), max_num_distractors)
        distractor_index = np.random.choice(distractor_index, num_random, replace=False)
        distractor_index = distractor_index.reshape(-1).tolist()

        result[query_index] = {
            "distractor_index": distractor_index,
            "distractor_class": distractor_class,
            "query_index": query_index,
            "query_class": query_class,
        }

    return result

@torch.no_grad()
def process_dataset(model, dataloader, device):
    """
    Process a dataset using a pre-trained classification model.
    We return the spatial feature representations, the predictions,
    the targets, top-1 accuracy and a confusion matrix as a dictionary.
    """
    top1 = torchmetrics.Accuracy(top_k=1)

    top1.to(device)
    model.to(device)

    model.eval()

    features = []
    preds = []
    targets = []

    for batch in dataloader:
        if isinstance(batch, dict):
            images, target = batch["image"].to(device), batch["target"].to(device)
        else:
            images, target = [t.to(device) for t in batch]

        if isinstance(model, PrototypeChooser):
            prob, min_distances, proto_presence, activations = model(images, gumbel_scale=10e3)
            bkb_fts = model.features(images)
            output = dict(
                logits = prob,
                features=bkb_fts
            )

        else:
            output = model(images)
        top1(output["logits"], target)

        features.append(output["features"].cpu())
        targets.append(target.cpu())
        preds.append(torch.argmax(output["logits"], dim=1).cpu())

    features = torch.cat(features, dim=0)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    confusion_mat = confusion_matrix(targets.numpy(), preds.numpy())
    print("top1", top1.compute())
    # assert False, f"{features.shape=}"
    result = {
        "features": features,
        "preds": preds,
        "targets": targets,
        "top1": top1.compute(),
        "confusion_matrix": confusion_mat,
    }


    return result
