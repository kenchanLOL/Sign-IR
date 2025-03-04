# *torch
from pickletools import optimize

# from sched import scheduler
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right

# *user-defined
from models import gloss_free_model
from datasets import S2T_Dataset
import utils as utils

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import test as test
import wandb
import copy
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger

from hpman.m import _
import hpargparse

# *metric
from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER

# try:
#     from nlgeval import compute_metrics
# except:
#     print('Please install nlgeval package.')

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import NativeScaler

# global definition
from definition import *

import signcl as signcl

cl_criterion = signcl.SignCL()


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Gloss-free Sign Language Translation script", add_help=False
    )
    parser.add_argument("--batch-size", default=16, type=int)

    # * distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--local_rank", default=0, type=int)

    # * advance params
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    # parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    # parser.add_argument(
    #     "--dist-eval",
    #     action="store_true",
    #     default=False,
    #     help="Enabling distributed evaluation",
    # )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--config", type=str, default="./configs/config_gloss_free_dsl.yaml"
    )

    # * data process params
    parser.add_argument("--input-size", default=224, type=int)
    parser.add_argument("--resize", default=256, type=int)

    # * visualization
    parser.add_argument("--attn_visualize", action="store_true")
    parser.add_argument("--tsne_visualize", action="store_true")

    return parser


def main(args, config):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating dataset:")
    tokenizer = MBartTokenizer.from_pretrained(config["model"]["transformer"])

    # train_data = S2T_Dataset(
    #     path=config["data"]["train_label_path"],
    #     tokenizer=tokenizer,
    #     config=config,
    #     args=args,
    #     phase="train",
    #     reg_label_path=config["data"]["dev_reg_label_path"]
        
    # )
    # print("train dataset", train_data)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_data, shuffle=True
    # )
    # train_dataloader = DataLoader(
    #     train_data,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     collate_fn=train_data.collate_fn,
    #     sampler=train_sampler,
    #     pin_memory=args.pin_mem,
    # )

    dev_data = S2T_Dataset(
        path=config["data"]["dev_label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="dev",
        reg_label_path=config["data"]["dev_reg_label_path"]
    )
    print("dev dataset", dev_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(
        dev_data, shuffle=False
    )
    dev_dataloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler,
        pin_memory=args.pin_mem,
        
    )

    test_data = S2T_Dataset(
        path=config["data"]["test_label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="test",
        reg_label_path=config["data"]["test_reg_label_path"]
    )
    print("test dataset", test_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data, shuffle=False
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        sampler=test_sampler,
        pin_memory=args.pin_mem,
    )

    print(f"Creating model:")
    tokenizer = MBartTokenizer.from_pretrained(
        config["model"]["transformer"], src_lang="de_DE", tgt_lang="de_DE"
    )
    model = gloss_free_model(config, args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    output_dir = Path(args.output_dir)
    if args.resume:
        print("Resuming Model Parameters... ")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)

    else:
        logger.warning(
            "Please specify the trained model: --resume /path/to/best_checkpoint.pth"
        )

    test_stats = evaluate(
        args,
        dev_dataloader,
        model,
        model_without_ddp,
        tokenizer,
        criterion,
        config,
        UNK_IDX,
        SPECIAL_SYMBOLS,
        PAD_IDX,
        device,
    )
    print(
        f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} "
    )
    test_stats = evaluate(
        args,
        test_dataloader,
        model,
        model_without_ddp,
        tokenizer,
        criterion,
        config,
        UNK_IDX,
        SPECIAL_SYMBOLS,
        PAD_IDX,
        device,
    )
    print(
        f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}"
    )
    return

    
def evaluate(
    args,
    dev_dataloader,
    model,
    model_without_ddp,
    tokenizer,
    criterion,
    config,
    UNK_IDX,
    SPECIAL_SYMBOLS,
    PAD_IDX,
    device,
    epoch=999,
    cl_decay=1.0,
):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        if args.tsne_visualize:
            frame_feats_by_recognition_label = {}
        for step, (src_input, tgt_input, _) in enumerate(
            metric_logger.log_every(dev_dataloader, 10, header)
        ):
            out_logits, output, frames_feature = model(src_input, tgt_input)
            total_loss = 0.0
            total_cl_loss = 0.0
            label = tgt_input["input_ids"].reshape(-1)

            logits = out_logits.reshape(-1, out_logits.shape[-1])
            tgt_loss = criterion(logits, label.to(device))

            total_loss += tgt_loss

            margin = (
                max(
                    10,
                    int(
                        (frames_feature.shape[1] // tgt_input["input_ids"].shape[1] + 1)
                        * 2.3
                    ),
                )
                * 2
            )
            num_negative = 30
            margin = min(
                margin, int((frames_feature.shape[1] - num_negative) / 2)
            )  # ensure num_frames margin for negative sampling
            cl_loss = cl_criterion(frames_feature, margin=margin)

            total_cl_loss += cl_loss

            metric_logger.update(loss=total_loss.item())
            metric_logger.update(cl_loss=total_cl_loss.item())
            metric_logger.update(cl_decay=cl_decay)
            output = model_without_ddp.generate(
                src_input,
                max_new_tokens=150,
                num_beams=4,
                decoder_start_token_id=tokenizer.lang_code_to_id["de_DE"],
            )

            tgt_input["input_ids"] = tgt_input["input_ids"].to(device)
            for i in range(len(output)):
                tgt_pres.append(output[i, :])
                tgt_refs.append(tgt_input["input_ids"][i, :])
            if args.tsne_visualize:
                all_frame_feats = frames_feature.reshape(-1, frames_feature.shape[-1]).cuda()
                all_labels = []
                for l in src_input["recognition_label"]:
                    all_labels += l
                # import pdb;pdb.set_trace()
                # print(all_frame_feats.shape, all_labels)
                # assert frames_feature.shape[0] == len(all_labels)
                for idx, l in enumerate(all_labels):
                    if l == "<PAD>":
                        continue
                    elif l not in frame_feats_by_recognition_label:
                        frame_feats_by_recognition_label[l] = torch.Tensor(all_frame_feats[idx].unsqueeze(0))
                    else:
                        frame_feats_by_recognition_label[l] = torch.cat(
                            (
                                frame_feats_by_recognition_label[l],
                                all_frame_feats[idx].unsqueeze(0),
                            ),
                            dim=0,
                        )
        if args.tsne_visualize:
            # utils.tsne_visualize(frame_feats_by_recognition_label, save_dir=args.output_dir)
            utils.tsne_visualize_topk(frame_feats_by_recognition_label, save_dir=args.output_dir)
            # import pickle
            # base_filename = "frame_features"
            # extension = ".pickle"
            # i = 1
            # while os.path.exists(os.path.join(args.output_dir, f"{base_filename}_{i}{extension}")):
            #     i += 1
            # save_path = os.path.join(args.output_dir, f"{base_filename}_{i}{extension}")
            # with open(save_path, 'wb') as f:
            #     pickle.dump(frame_feats_by_recognition_label, f)
            utils.calculate_sdr(frame_feats_by_recognition_label, save_dir=args.output_dir)
        # if (step + 1) % 10 == 0 and args.visualize and utils.is_main_process():
        #     utils.visualization(model_without_ddp.visualize())

    pad_tensor = torch.ones(200 - len(tgt_pres[0])).to(device)
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)
    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=PAD_IDX)

    pad_tensor = torch.ones(200 - len(tgt_refs[0])).to(device)
    tgt_refs[0] = torch.cat((tgt_refs[0], pad_tensor.long()), dim=0)
    tgt_refs = pad_sequence(tgt_refs, batch_first=True, padding_value=PAD_IDX)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)

    bleu = BLEU()
    bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score

    metric_logger.meters["belu4"].update(bleu_s)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* BELU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.belu4, losses=metric_logger.loss
        )
    )

    os.makedirs(name=args.output_dir + "/output", exist_ok=True)
    with open(args.output_dir + f"/output/tmp_pres_{epoch}.txt", "w") as f:
        for i in range(len(tgt_pres)):
            f.write(tgt_pres[i] + "\n")

    if utils.is_main_process() and utils.get_world_size() == 1:
        with open(args.output_dir + "/tmp_pres.txt", "w") as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i] + "\n")
        with open(args.output_dir + "/tmp_refs.txt", "w") as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i] + "\n")
        print("\n" + "*" * 80)
        # metrics_dict = compute_metrics(hypothesis=args.output_dir + '/tmp_pres.txt',
        #                                references=[args.output_dir + '/tmp_refs.txt'], no_skipthoughts=True)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        "Gloss-free Sign Language Translation script", parents=[get_args_parser()]
    )
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ["WANDB_MODE"] = (
        config["training"]["wandb"]
    )
    if utils.is_main_process():
        wandb.init(project="GF-SLT", config=config)
        wandb.run.name = args.output_dir.split("/")[-1]
        wandb.define_metric("epoch")
        wandb.define_metric("training/*", step_metric="epoch")
        wandb.define_metric("dev/*", step_metric="epoch")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
