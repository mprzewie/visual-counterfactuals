# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import albumentations
import torchvision.transforms
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Subset, TensorDataset
import torchvision.transforms as transforms

from PIL import  Image
import model.auxiliary_model as auxiliary_model
import numpy as np
import torch
import yaml

from explainer.counterfactuals import compute_counterfactual
from explainer.eval import compute_eval_metrics
from explainer.utils import get_query_distractor_pairs_with_predetermined_targets, process_dataset
from tqdm import tqdm

from protopool import PrototypeChooser
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
import pickle
from pathlib import Path as PathlibPath

parser = argparse.ArgumentParser(description="Generate counterfactual explanations")
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--dataset", type=PathlibPath, required=True)



def fadeout_cell(h: int, w: int, deg=1) -> np.ndarray:
    cell = np.ones((h, w))
    h2, w2 = h / 2, w / 2

    r = min(h2, w2)
    for y in range(h):
        for x in range(w):
            yd, xd = np.abs(y - h2), np.abs(x - w2)
            d = np.sqrt(yd ** 2 + xd ** 2)

            v = (1 - ((r - d) / r) ** 2) if d < r else 1
            cell[y][x] = v ** deg

    cell = cell - cell.min()
    cell = cell / (cell.max() - cell.min())
    return cell


def paste(
        im1,
        im2,
        box,
        mask2
):
    im1arr = np.array(im1)
    im2arr = np.array(im2)
    h2, w2, c = im2arr.shape
    bw1, bh1, bw2, bh2 = box
    assert (h2, w2) == (mask2.shape) == (bh2 - bh1, bw2 - bw1), ((h2, w2), (mask2.shape), (bh2 - bh1, bw2 - bw1))
    h1, w1, c = im1arr.shape

    if any([
        bw1 > w1,
        bw2 < 0,
        bh1 > h1,
        bh2 < 0
    ]):
        return im1.copy()

    mask1 = np.ones((h1, w1))

    #### truncate box, im2 and m2 so that they're not out of bounds
    print("pre", box, mask2.shape)

    bw1o = max(-bw1, 0)
    bw2o = min(w1 - bw2, 0)
    bh1o = max(-bh1, 0)
    bh2o = min(h1 - bh2, 0)

    im2arr = im2arr[bh1o:(h2 + bh2o), bw1o:(w2 + bw2o)]
    mask2 = mask2[bh1o:(h2 + bh2o), bw1o:(w2 + bw2o)]
    box = (bw1 + bw1o, bh1 + bh1o, bw2 + bw2o, bh2 + bh2o)

    #### sanity check

    h2, w2, c = im2arr.shape
    bw1, bh1, bw2, bh2 = box
    assert (h2, w2) == (mask2.shape) == (bh2 - bh1, bw2 - bw1), ((h2, w2), (mask2.shape), (bh2 - bh1, bw2 - bw1))

    ####

    mask1[bh1:bh2, bw1:bw2] = mask2

    im2p = np.zeros_like(im1arr)
    im2p[bh1:bh2, bw1:bw2] = im2arr
    mask1 = mask1.reshape((h1, w1, 1))

    arr = (im1arr * mask1) + (im2p * (1 - mask1))
    return Image.fromarray(arr.astype("uint8"))



def main():
    args = parser.parse_args()

    # parse args
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)

    if "model_type" not in config:
        print("No model type specified, setting classifier")
        config["model_type"] = "classifier"

    experiment_name = f"{os.path.basename(args.config_path).split('.')[0]}_opt_records"
    dirpath = os.path.join(Path.output_root_dir(), experiment_name)
    os.makedirs(dirpath, exist_ok=True)

    # create dataset

    # we use dataset as the path to opt_records from plugen

    with args.dataset.open("rb") as f:
        plugen_opt_records = pickle.load(f)

    images= []
    labels = []
    classes_to = []
    
    # assert False, { 
    #     k: (
    #         v.shape if isinstance(v, np.ndarray)
    #         else type(v)
    #     ) for (k,v) in plugen_opt_records[0].items() 
    # }
    
    # print()
    
    for optr in plugen_opt_records:
        images.extend(optr["images"][0])
        labels.extend(optr["class_from"])
        classes_to.extend(optr["class_to"])
        

    test_transform = albumentations.Compose(
        [
            albumentations.Resize(224, 224),
            albumentations.Normalize((0.471, 0.460, 0.454), (0.267, 0.266, 0.271)),
            ToTensorV2(),
        ],
    )

    test_transform_wo_normalize= albumentations.Compose(
        [
            albumentations.Resize(224, 224),
            ToTensorV2(),
        ]
    )

    images_trans = torch.stack([test_transform(image=i)["image"] for i in images])
    images_trans_wo_normalize = torch.stack([test_transform_wo_normalize(image=i)["image"] for i in images])

    labels = torch.tensor(labels)

    dataset = TensorDataset(images_trans, labels)
    dataset_resize_only =  TensorDataset(images_trans_wo_normalize, labels)

    dataloader = get_test_dataloader(config, dataset)
    
    
    gt_dataset_transform, gt_dataset_transform_wo_normalize = (get_test_transform_resize(), get_test_transform_resize_wo_normalize())
    gt_dataset = get_test_dataset_framed(transform=gt_dataset_transform, return_image_only=True)
    gt_dataset_resize_only = get_test_dataset_framed(transform=gt_dataset_transform_wo_normalize)
    
    gt_dataloader = get_test_dataloader(config, gt_dataset)
    
    # device
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    # load classifier
    print("Load " + config["model_type"] + " weights")
    model = get_model(config)
    model_path = os.path.join(
        Path.output_root_dir(),
        config["counterfactuals_kwargs"]["model"],
    )
    if config["model_type"] == "classifier":
        state_dict = torch.load(model_path)["state_dict"]
        for key in list(state_dict.keys()):
            state_dict[key[len("model.") :]] = state_dict[key]
            del state_dict[key]
        model.load_state_dict(state_dict, strict=True)
    elif config["model_type"] == "protopool":
        model.load_state_dict(
            torch.load(config["counterfactuals_kwargs"]["model"], map_location='cpu')['model_state_dict'], strict=True
        )


    with torch.no_grad():
        # process dataset
        print("Pre-compute classifier predictions")
        result = process_dataset(model, dataloader, device)
        features = result["features"]
        preds = result["preds"].numpy()
        targets = result["targets"].numpy()
        print("Query Top-1 accuracy: {:.2f}".format(100 * result["top1"]))


        print("Preprocess ground-truth dataset")
        gt_result = process_dataset(model, gt_dataloader, device)
        gt_features = gt_result["features"]
        gt_preds = gt_result["preds"].numpy()
        gt_targets = gt_result["targets"].numpy()
        print("GT Top-1 accuracy: {:.2f}".format(100 * gt_result["top1"]))
    
    
    for c in set(classes_to):
        
        ct = [i for i,t in enumerate(gt_targets) if t==c]
        cok = [i for i, (t,p) in enumerate(zip(gt_targets, gt_preds)) if t==c and t==p]
        nct = len(ct)
        ncok = len(cok)
        
        print(c, f"{(ncok / nct):.2f}", ncok, nct)
        print("targets", ct)
        print("predictions", cok)
        print("---")
        
            
    print("Pre-compute query-distractor pairs")
    

    query_distractor_pairs = get_query_distractor_pairs_with_predetermined_targets(
        query_dataset = dataset,
        targets=classes_to,
        ground_truth_dataset=gt_dataset,
        max_num_distractors=config["counterfactuals_kwargs"]["max_num_distractors"]
    )
    
    
    # get classifier head
    classifier_head = model.get_classifier_head() if not isinstance(model, PrototypeChooser) else model.post_backbone_classification()
    classifier_head = torch.nn.DataParallel(classifier_head.cuda())
    classifier_head.eval()

    # auxiliary features for soft constraint
    if config["counterfactuals_kwargs"]["apply_soft_constraint"]:
        print("Pre-compute auxiliary features for soft constraint")
        aux_model, aux_dim, n_pix = auxiliary_model.get_auxiliary_model()
        aux_transform = transforms.Compose(
            [
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        images_trans_aux = torch.stack([aux_transform(i) for i in images])
        aux_dataset = TensorDataset(images_trans_aux)
        
        aux_loader = get_test_dataloader(config, aux_dataset)

        auxiliary_features = auxiliary_model.process_dataset(
            aux_model,
            aux_dim,
            n_pix,
            aux_loader,
            device,
        ).numpy()
        
        gt_aux_transform = get_imagenet_test_transform()
        gt_aux_dataset = get_test_dataset_framed(transform=gt_aux_transform, return_image_only=True)
        gt_aux_loader = get_test_dataloader(config, gt_aux_dataset)
        gt_auxiliary_features = auxiliary_model.process_dataset(
            aux_model,
            aux_dim,
            n_pix,
            gt_aux_loader,
            device,
        ).numpy()
        
        
        use_auxiliary_features = True
        
    else:
        use_auxiliary_features = False
    
    
    # compute counterfactuals
    print("Compute counterfactuals")
    counterfactuals = {}
    
    opt_records = []
    # n_failed_counterfactuals = 0
    for query_index in tqdm(range(len(dataset))):
        if query_index not in query_distractor_pairs.keys():
            continue  # skips images that were classified incorrectly

        # gather query features
        query_fm = features[query_index]  # dim x n_row x n_row
        query_pred = preds[query_index]
        # if query_pred != targets[query_index]:
        #     assert False, "We do not do that here"
        #     continue  # skip if query classified incorrect

        # gather distractor features
        distractor_target = query_distractor_pairs[query_index][
            "distractor_class"
        ]  # noqa
        distractor_index = query_distractor_pairs[query_index][
            "distractor_index"
        ]  # noqa
        
        
        if isinstance(distractor_index, int):
            assert False, "We do not do that here"
            if preds[distractor_index] != distractor_target:
                continue  # skip if distractor classified is incorrect
            distractor_index = [distractor_index]

        else:  # list
            di = len(distractor_index)
            
            distractor_index = [
                jj for jj in distractor_index if gt_preds[jj] == distractor_target
            ]

            if len(distractor_index) == 0:
                assert False, "no distractors to use, should we assign accuracy=0?"
                # n_failed_counterfactuals += 1
                # continue  # skip if no distractors classified correct
            
        distractor_fm = torch.stack([gt_features[jj] for jj in distractor_index], dim=0)

        # soft constraint uses auxiliary features
        if use_auxiliary_features:
            query_aux_features = torch.from_numpy(
                auxiliary_features[query_index]
            )  # aux_dim x n_row x n_row
            
            distractor_aux_features = torch.stack(
                [torch.from_numpy(gt_auxiliary_features[jj]) for jj in distractor_index],
                dim=0,
            )  # n x aux_dim x n_row x n_row

        else:
            query_aux_features = None
            distractor_aux_features = None

        # compute counterfactual
        list_of_edits, list_of_preds = compute_counterfactual(
            query=query_fm,
            distractor=distractor_fm,
            classification_head=classifier_head,
            distractor_class=distractor_target,
            query_aux_features=query_aux_features,
            distractor_aux_features=distractor_aux_features,
            lambd=config["counterfactuals_kwargs"]["lambd"],
            temperature=config["counterfactuals_kwargs"]["temperature"],
            topk=config["counterfactuals_kwargs"]["topk"]
            if "topk" in config["counterfactuals_kwargs"].keys()
            else None,
        )

        list_of_edits = list_of_edits[:config["counterfactuals_kwargs"].get("max_edits", 1000)]
        list_of_preds = list_of_preds[1:(1+config["counterfactuals_kwargs"].get("max_edits", 1000))]
        
        # print(query_pred, distractor_target)
        # print(list_of_edits)
        # print(list_of_preds)

        # assert len(distractor_index)==1, len(distractor_index)
        # di = list(distractor_index)[0]


        # print(query_index, distractor_index)

        q_img, query_label = dataset_resize_only[query_index]
        # q_img = "image"]
        # # q_parts = dataset[query_index]["parts"]
        # query_label = dataset[query_index]["target"]

        c, h, w = q_img.shape
        fc, fh, fw = query_fm.shape

        cell_h, cell_w = h // fh, w // fw

        cell_mul = 3
        assert cell_mul % 2 == 1
        cell_w_o = int(cell_w * ((cell_mul -1) / 2))
        cell_h_o = int(cell_h * ((cell_mul -1) / 2))
        fcell_mul = fadeout_cell(cell_h * cell_mul, cell_w * cell_mul, deg=10)
        fcell_bkp = fadeout_cell(cell_h, cell_w, deg=10)



        d_imgs: List[Image.Image] = [
            ttf.to_pil_image(gt_dataset_resize_only[di]["image"]).convert("RGBA")
            for di in distractor_index
        ]
        d_labels = [gt_dataset_resize_only[di]["target"] for di in distractor_index]


        fig = plt.figure(constrained_layout=True, figsize=(30, 6))
        subfigs = fig.subfigures(nrows=1, ncols=5, )

        # fig, ax = plt.subplot)


        q_img_pil: Image.Image = ttf.to_pil_image(q_img).convert("RGBA")

        ax0 = subfigs[0].subplots()
        ax0.imshow(q_img_pil)
        ax0.set_title(f"{query_label=}")

        n_d_s = int(np.ceil(np.sqrt(len(d_imgs))))

        ax1 = subfigs[1].subplots(nrows=n_d_s, ncols=n_d_s)
        if n_d_s==1:
            ax1 = [[ax1]]

        [[a.axis("off") for a in a_] for a_ in ax1]
        for di, d_img_pil in enumerate(d_imgs):
            dr = di // n_d_s
            dc = di % n_d_s
            ax1[dr][dc].imshow(d_img_pil)

        # subfigs[1].suptitle(f"{distractor_label=}")


        q_patches = Image.new("RGBA", (w, h))
        d_patches = Image.new("RGBA", (w, h))

        counterfactual=q_img_pil.copy()
        print("wh", (w,h))
        for (e_q, e_d) in list_of_edits:
            e_q_h = (e_q // fw) * cell_h
            e_q_w = (e_q % fw) * cell_w

            e_d_i = e_d // (fw*fw)
            e_d_c = e_d % (fw*fw)
            e_d_h = (e_d_c // fw) *cell_h
            e_d_w = (e_d_c % fw) * cell_w

            print(f"{(e_q, e_d)=} => {((e_q_h, e_q_w), (e_d_i, e_d_h, e_d_w))}")

            q_box = (e_q_w-cell_w_o, e_q_h-cell_h_o, e_q_w + cell_w + cell_w_o, e_q_h + cell_h + cell_h_o)

            d_box = (e_d_w-cell_w_o, e_d_h-cell_h_o, e_d_w +  cell_w + cell_w_o, e_d_h + cell_h + cell_h_o)

            use_fcell = fcell_mul
            if any([
                d_box[0] < 0,
                d_box[1] < 0,
                d_box[2] > w,
                d_box[3] > h
            ]):
                # print("----")
                # print((w,h))
                # print(d_box)
                # print("----")

                # if distractor cell is out of bounds, remove the pretty overlap
                q_box = (e_q_w, e_q_h, e_q_w + cell_w, e_q_h + cell_h)
                d_box = (e_d_w, e_d_h, e_d_w + cell_w, e_d_h + cell_h)
                use_fcell = fcell_bkp

            q_crop = q_img_pil.crop(q_box)

            # print("q_box", q_box)
            # print("d_box", d_box)
            d_crop = d_imgs[e_d_i].crop(d_box)

            q_patches.paste(q_crop, q_box)
            d_patches.paste(d_crop, d_box)

            # d_box_2
            counterfactual = paste(
                counterfactual,
                d_crop,
                q_box,
                use_fcell
            )


        ax2 = subfigs[2].subplots()
        ax2.imshow(q_patches)
        ax2.set_title(f"query edits,  #edits = {len(list_of_edits)}")

        ax3 = subfigs[3].subplots()
        ax3.imshow(d_patches)
        ax3.set_title("distractor edits")

        ax4 = subfigs[4].subplots()
        ax4.imshow(counterfactual)
        ax4.set_title(f"c-factual")
        fig.suptitle(
            f"q{query_index} / d{distractor_index}: {query_pred} -> {distractor_target} | {','.join([str(p) for p in list_of_preds])}"
        )
        img_path = PathlibPath(dirpath) / f"cf_{query_index}.png"

        img_path.parent.mkdir(exist_ok=True, parents=True)
        print(img_path)
        fig.savefig(img_path)
        


        counterfactuals[query_index] = {
            "query_index": query_index,
            "distractor_index": distractor_index,
            "query_target": query_pred,
            "distractor_target": distractor_target,
            "edits": list_of_edits,
        }
        
        
        q_np = np.array(q_img_pil.convert("RGB"))
        cf_np = np.array(counterfactual.convert("RGB"))
        img_diff = ((q_np - cf_np) ** 2).mean()
        accuracy = 1 if (query_pred == distractor_target) else float(distractor_target == list_of_preds[-1])
        
        if len(list_of_preds) == 0:
            assert query_pred == distractor_target
            list_of_preds = [distractor_target]

        opt_record = {
            "class_from": [query_pred],
            "class_to": [distractor_target],
            "y_pred": [[query_pred], [list_of_preds[-1]]],
            "iterations": [0, len(list_of_preds)],
            "images": [[q_np], [cf_np]],
            "img_diff": [[0], [img_diff]],
            "accuracy": [[lop==distractor_target] for lop in [query_pred, list_of_preds[-1]]]
        }
        opt_record = {
            k: np.array(v)
            for (k,v) in opt_record.items()
        }
        opt_record["best"] = {k: opt_record[k] for k in ["img_diff", "accuracy"]}
        opt_records.append(opt_record)
    
    # save result
    np.save(os.path.join(dirpath, "counterfactuals.npy"), counterfactuals)
    with (PathlibPath(dirpath) / "opt_records.pkl").open("wb") as f:
        pickle.dump(opt_records, f)


    # evaluation
    print("Generated {} counterfactual explanations".format(len(counterfactuals)))
    average_num_edits = np.mean([len(res["edits"]) for res in counterfactuals.values()])
    print("Average number of edits is {:.2f}".format(average_num_edits))
    # print("n failures", n_failed_counterfactuals)

    # result = compute_eval_metrics(
    #     counterfactuals,
    #     dataset=dataset,
    # )

#     print("Eval results single edit: {}".format(result["single_edit"]))
#     print("Eval results all edits: {}".format(result["all_edit"]))


if __name__ == "__main__":
    main()
