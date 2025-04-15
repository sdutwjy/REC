import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 设置PyTorch内存管理
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
import cv2
import numpy as np
import math
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from groundingdino.util.base_api import load_model, threshold
import os
import numpy as np
from datetime import datetime
import random

from utils.processor import DataProcessor
from utils.criterion import SetCriterion
from utils.image_loader import get_loader

from clearml import Task
from argparse import ArgumentParser

# 添加torch.cuda.amp支持混合精度训练
from torch.cuda.amp import autocast, GradScaler

class CustomClearML():
    def __init__(self, project_name, task_name):
        self.task = Task.init(project_name, task_name)
        self.logger = self.task.get_logger()

    def __call__(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value, iteration)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
parser = ArgumentParser()
parser.add_argument('--stats_dir', type=str, default='/scratch/jianyong/REC/stats/debug')
parser.add_argument('--text_threshold', type=float, default=0.25)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test', action='store_true')
parser.add_argument('--vis', action='store_true')
# 添加内存管理相关参数
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--fp16', action='store_true', help='use mixed precision training')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
args = parser.parse_args()

set_seed(args.seed)

task_name='_'.join(args.stats_dir.split('/')[-3:])
       
# task_name='share_qformer_moe_debug'
clearml_logger = CustomClearML('REC', task_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


""" data """
processor = DataProcessor()
annotations = processor.annotations

BATCH_SIZE = args.batch_size
train_loader = get_loader(processor, 'train', BATCH_SIZE)
val_loader = get_loader(processor, 'val', BATCH_SIZE)
test_loader = get_loader(processor, 'test', BATCH_SIZE)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
print("Data loaded!")
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")


""" model"""
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "pretrained/groundingdino_swint_ogc.pth"
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
model = model.to(device)

# freeze encoders
for param in model.backbone.parameters(): 
    param.requires_grad = False
for param in model.bert.parameters():
    param.requires_grad = False


""" criterion """
criterion = SetCriterion()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0001)

# 添加混合精度训练
scaler = GradScaler() if args.fp16 else None


def train(epoch):
    print(f"Training on train set data")
    model.train()
    loader = loaders['train']

    train_mae = 0
    train_rmse = 0
    
    train_tp = 0
    train_fp = 0
    train_fn = 0
    
    counter = 0
    counter_for_image = 0
    train_size = len(loader.dataset) 

    # 梯度累积设置
    optimizer.zero_grad()
    for batch_idx, (images, captions, shapes, img_caps) in enumerate(loader): # tensor, list of list [caption] for each image in the batch, list, list of list [(img, cap)] for each img in the batch
        # images: [b1_img, b2_img,...] captions: [ [b1_cap1, b1_cap2], [b2_cap1, b2_cap2], ...]

        mask_bi = [i for i, img_cap_list in enumerate(img_caps) for _ in img_cap_list] # index for each img,cap pair in the batch
        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list] 
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]
        
        # duplicate each image number of times that is equal to the number of captions for that image
        images = torch.stack([images[i] for i, caption_list in enumerate(captions) for _ in caption_list], dim=0)
        captions = [caption for caption_list in captions for caption in caption_list] # flatten list of list
        images = images.to(device)
        
        # 使用混合精度训练
        with autocast(enabled=args.fp16):
            outputs = model(images, captions=captions)
            
            outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
            
            # prepare targets
            emb_size = outputs["pred_logits"].shape[2]
            targets = prepare_targets(anno_b, captions, shapes, emb_size)

            loss_dict = criterion(outputs, targets, mask_bi)
            weight_dict = criterion.weight_dict

            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # 根据梯度累积步骤缩放损失
            loss = loss / args.gradient_accumulation_steps
        
        # 使用混合精度反向传播
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 如果达到梯度累积步数或是最后一个批次
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(loader) - 1:
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        counter_for_image += 1
        results = threshold(outputs, captions, model.tokenizer, text_threshold = 0.25)
        for b in range(len(results)): # (bs*num_cap)
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()

            points = [[box[0], box[1]] for box in boxes] # center points

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            train_mae += cnt_err
            train_rmse += cnt_err ** 2

            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            train_tp += TP
            train_fp += FP
            train_fn += FN
        
            counter += 1
            
            print(f'[train] ep {epoch} ({counter_for_image}/{train_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(train_mae/counter)}, RMSE: {int((train_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')
            
        # 释放显存
        torch.cuda.empty_cache()
        
    
    train_mae = train_mae / counter
    train_rmse = (train_rmse / counter) ** 0.5

    train_precision = train_tp / (train_tp + train_fp) if train_tp + train_fp != 0 else 0.0
    train_recall = train_tp / (train_tp + train_fn) if train_tp + train_fn != 0 else 0.0
    train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if train_precision + train_recall != 0 else 0.0
    
    torch.cuda.empty_cache()

    return train_mae, train_rmse, train_tp, train_fp, train_fn, train_precision, train_recall, train_f1




def eval(split, epoch=None):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]

    eval_mae = 0
    eval_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0
    
    counter = 0
    counter_for_image = 0
    eval_size = len(loader.dataset)

    for images, captions, shapes, img_caps in loader: # tensor, list, list, list

        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list] 
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]

        
        images = torch.stack([images[i] for i, caption_list in enumerate(captions) for _ in caption_list], dim=0)
        captions = [caption for caption_list in captions for caption in caption_list] # flatten list of list
        images = images.to(device)
        with torch.no_grad():
            # 评估时也使用混合精度以节省内存
            with autocast(enabled=args.fp16):
                outputs = model(images, captions=captions)

        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 

        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(anno_b, captions, shapes, emb_size)


        counter_for_image += 1
        
        results = threshold(outputs, captions, model.tokenizer, text_threshold = 0.25)
        for b in range(len(results)):
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()


            points = [[box[0], box[1]] for box in boxes]

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            
            # ------------------------------------------------------------------------------------------------------------------------
            if args.vis:
                img_size = np.array(shapes[b])[::-1]  # [W, H]
                if len(points):
                    pred = np.round(np.array(points) * img_size, 2)   # list -> array
                else:
                    pred = np.array([])
                if len(targets[b]["points"]):
                    gt = np.round(targets[b]["points"].detach().cpu().numpy() * img_size, 2)  # tensor -> array
                else:
                    gt = np.array([])

                pred_map = fidt_generate(pred, img_size)  # [H, W, C]
                gt_map = fidt_generate(gt, img_size)
                
                img_map = vis_norm(denormalize_tensor(images[b]).cpu().numpy().transpose(1, 2, 0)[:, :, ::-1])  # RGB -> BGR
                img_map = cv2.resize(img_map, img_size)  # [H, W, C] [W, H]

                alpha = 0.2
                img_add_gt = cv2.addWeighted(img_map, alpha, gt_map, 1 - alpha, 0)
                img_add_pred = cv2.addWeighted(img_map, alpha, pred_map, 1 - alpha, 0)
                combined_image = cv2.hconcat([img_map, img_add_pred, img_add_gt])

                gt_save_name = os.path.splitext(img_caps[b][0])[0] + "_" + img_caps[b][1].replace(" ", "_") + "_" + str(gt_cnt) + "_gt" + os.path.splitext(img_caps[b][0])[1]
                pred_save_name = os.path.splitext(img_caps[b][0])[0] + "_" + img_caps[b][1].replace(" ", "_") + "_" + str(pred_cnt) + "_pred" + os.path.splitext(img_caps[b][0])[1]
                cat_save_name = os.path.splitext(img_caps[b][0])[0] + "_" + img_caps[b][1].replace(" ", "_") + "_" + str(pred_cnt) + "_pred" + "_" + str(gt_cnt) + "_gt" + os.path.splitext(img_caps[b][0])[1]
                
                save_dir = os.path.join(stats_dir, "vis")
                os.makedirs(save_dir, exist_ok=True)

                cat_dir = os.path.join(save_dir, "cat")
                os.makedirs(cat_dir, exist_ok=True)
                
                cv2.imwrite(f"{save_dir}/{gt_save_name}", img_add_gt)
                cv2.imwrite(f"{save_dir}/{pred_save_name}", img_add_pred)
                cv2.imwrite(f"{cat_dir}/{cat_save_name}", combined_image)
            # ------------------------------------------------------------------------------------------------------------------------
            
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2
        
            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN


            counter += 1
            
            print(f'[{split}] ep {epoch} ({counter_for_image}/{eval_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(eval_mae/counter)}, RMSE: {int((eval_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')
        
        # 释放显存
        torch.cuda.empty_cache()
        

    eval_mae = eval_mae / counter
    eval_rmse = (eval_rmse / counter) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0
    
    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1


def denormalize_tensor(tensor):
    mean = (
        torch.tensor([0.485, 0.456, 0.406])
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(tensor.device)
    )
    std = (
        torch.tensor([0.229, 0.224, 0.225])
        .unsqueeze(-1)
        .unsqueeze(-1)
        .to(tensor.device)
    )
    denormalized_tensor = tensor * std + mean
    return denormalized_tensor

def vis_norm(var):
    var = (var - np.min(var)) / (np.max(var) - np.min(var))
    var = var * 255
    var = var.astype(np.uint8)

    return var

def fidt_generate(gt, size):
    width, height = tuple(size)
    d_map = (np.zeros([height, width]) + 255).astype(np.uint8)
    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])  # H
        y = np.max([1, math.floor(gt[o][0])])  # W
        if x >= height or y >= width:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0
    
    distance_map = distance_map / np.max(distance_map) * 255
    distance_map = distance_map.astype(np.uint8)
    distance_map = cv2.applyColorMap(distance_map, 2)

    return distance_map


def prepare_targets(anno_b, captions, shapes, emb_size):
    for anno in anno_b:
        if len(anno['points']) == 0:
            anno['points'] = [[0,0]]
    gt_points_b = [np.array(anno['points']) / np.array(shape)[::-1] for anno, shape in zip(anno_b, shapes)] # (h,w) -> (w,h)
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b] 

    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points] 

    
    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt")

    # find last index of special token (.)
    end_idxes = [torch.where(input_ids==1012)[0][-1] for input_ids in tokenized['input_ids']] 
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:,:end_idx] = 1.0 

    caption_sizes = [end_idx + 2 for end_idx in end_idxes]  # incl. CLS and SEP

    targets = [{"points": img_gt_points.to(device), "labels": img_gt_logits.to(device), "caption_size": caption_size} for img_gt_points, img_gt_logits, caption_size in zip(gt_points, gt_logits, caption_sizes)] 

    return targets


def distance_threshold_func(boxes): # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2]*box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas)//2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w**2 + h**2) / 2.0
    
    return threshold

def calc_loc_metric(pred_boxes, gt_points): # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0
    
    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
    # use Hungarian algorithm to find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # determine TP, FP, FN
    TP = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] < dist_threshold:
            TP += 1
    
    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    Precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.0
    
    return TP, FP, FN, Precision, Recall, F1


# main 

stats_dir = args.stats_dir
os.makedirs(stats_dir, exist_ok=True)

stats_file = f"{stats_dir}/stats.txt"
stats = list()

print(f"Saving stats to {stats_file}")

with open(stats_file, 'a') as f:
    header = ['train_mae', 'train_rmse', 'train_TP', 'train_FP', 'train_FN', 'train_precision', 'train_recall', 'train_f1', '||', 'val_mae', 'val_rmse', 'val_TP', 'val_FP', 'val_FN', 'val_precision', 'val_recall', 'val_f1', '||', 'test_mae', 'test_rmse', 'test_TP', 'test_FP', 'test_FN', 'test_precision', 'test_recall', 'test_f1']
    f.write("%s\n" % ' | '.join(header))


best_mae = float('inf')
best_model = None
for epoch in range(0, 20):

    train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1 = train(epoch)
    val_mae, val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1 = eval('val', epoch)
    
    if best_mae > val_mae:
        best_mae = val_mae
        print(f"New best F1: {best_mae}")
        best_model = copy.deepcopy(model)
    
    stats.append([train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1, "||", val_mae, val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1, "||", 0,0,0,0,0, 0,0,0])

    metric_dict = {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_TP": train_TP,
            "train_FP": train_FP,
            "train_FN": train_FN,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_TP": val_TP,
            "val_FP": val_FP,
            "val_FN": val_FN,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
        }
    for key, value in metric_dict.items():
        clearml_logger(key, key, value, epoch)
    
    with open(stats_file, 'a') as f:
        s = stats[-1]
        for i, x in enumerate(s):
            if type(x) != str:
                s[i] = str(round(x,4))
        f.write("%s\n" % ' | '.join(s))
    torch.cuda.empty_cache()

if best_model is not None:  # 确保只有在模型有更新时才保存
        model_name = f'{stats_dir}/model.pth'
        torch.save({"model": best_model.state_dict()}, model_name)
        print(f"Best model saved to {model_name}")
else:
        print("Best model not updated. No model saved.")
torch.cuda.empty_cache()


# Inference on test set
print(f"Inference on test set using best model: {model_name}")
model = load_model(CONFIG_PATH, model_name)
model = model.to(device)
test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1 = eval('test', -1)
print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}")
# write to stats file
line_inference = [0,0,0,0,0, 0,0,0, "||", 0,0,0,0,0, 0,0,0, "||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1]
with open(stats_file, 'a') as f:
    s = line_inference
    for i, x in enumerate(s):
        if type(x) != str:
            s[i] = str(round(x,4))
    f.write("%s\n" % ' | '.join(s))
torch.cuda.empty_cache()

