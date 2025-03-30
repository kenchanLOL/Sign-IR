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
from datasets_with_recognition_label import S2T_Dataset
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
# bin from GF_VLP
binned_labels = {'bin_1': ['WO', 'ICH', 'SELTEN', 'SECHSHUNDERT', 'DU', 'DAZWISCHEN', 'ALLE', 'VERAENDERN', 'WINTER', 'UNTER', 'HABEN', 'neg-HABEN', 'KURZ', 'GEFRIEREN', 'negalp-AUCH', 'MAI', 'ZEIGEN-BILDSCHIRM', 'VORSICHT', 'SEE', 'METER', 'ZONE', 'DURCHGEHEND', 'TAUEN', 'UNTERSCHIED', 'VORAUSSAGE', 'FEBRUAR'], 'bin_2': ['NAH', 'TATSAECHLICH', 'DARUM', 'WIE-IMMER', 'NOVEMBER', 'ALLGAEU', 'MACHEN', 'ERSTE', 'DAUER', 'DRITTE', 'ACHTE', 'WIEDER', 'TAGSUEBER', 'WAS', 'REST', 'WECHSEL', 'SCHEINEN', 'GLEICH', 'ZWEITE', 'GERADE', 'HABEN2', 'NULL', 'M', 'OFT', 'VORAUS', 'WENIGER'], 'bin_3': ['WENIG', 'negalp-KEIN', 'ABWECHSELN', 'JUNI', 'MITTWOCH', 'MITTEILEN', 'NACHMITTAG', 'LANGSAM', 'GRAUPEL', 'DREI', 'KOELN', 'BURG', 'INFORMIEREN', 'SKANDINAVIEN', 'EIN-PAAR-TAGE', 'FUENFTE', 'WOCHE', 'SINKEN', 'HIMMEL', 'SCHOTTLAND', 'BERLIN', 'FEUCHT', 'MAXIMAL', 'VOGEL', 'neg-REGEN', 'AUGUST'], 'bin_4': ['ZUERST', 'GRUND', 'WIE-AUSSEHEN', 'MITTAG', 'WEITER', 'AUFLOESEN', 'PLUS', 'LOCKER', 'SCHWER', 'BODEN', 'DIENSTAG', 'TSCHUESS', 'ZWEI', 'MAERZ', 'APRIL', 'OKTOBER', 'STERN', 'HOEHE', 'BESSER', 'DREISSIG', 'UNWETTER', 'WALD', 'JULI', 'LEICHT', 'VIERZEHN', 'HAGEL'], 'bin_5': ['VOR', 'FUENF', 'SIEBEN', 'DABEI', 'IN-KOMMEND', 'AEHNLICH', 'ORT', 'SPAETER', 'BAYERN', 'HERBST', 'KUEHL', 'VIER', 'STRASSE', 'FROST', 'ZWISCHEN', 'MAL', 'ACHT', 'MONTAG', 'BRAND', 'EINS', 'DOCH', 'WIE', 'DREIZEHN', 'SACHSEN', 'SAMSTAG', 'DRUCK'], 'bin_6': ['NAECHSTE', 'GLATT', 'RUHIG', 'SCHNEIEN', 'NEUNZEHN', 'SEPTEMBER', 'SECHSZEHN', 'ZWOELF', 'SIEBZEHN', 'MIT', 'KALT', 'SPEZIELL', 'WAHRSCHEINLICH', 'SUEDOST', 'UEBERWIEGEND', 'SUEDWEST', 'BLEIBEN', 'NEUN', 'DESHALB', 'TEIL', 'BEGRUESSEN', 'NORDOST', 'STARK', 'DEZEMBER', 'NACH', 'WENN'], 'bin_7': ['MEER', 'BIS', 'WUENSCHEN', 'DEUTSCHLAND', 'MINUS', 'MISCHUNG', 'VERSCHIEDEN', 'WECHSELHAFT', 'ZUSCHAUER', 'HAUPTSAECHLICH', 'GEFAHR', 'FUENFZEHN', 'SEHEN', 'DONNERSTAG', 'TRUEB', 'LUFT', 'ODER', 'ABER', 'FRUEH', 'DAZU', 'SO', 'UEBER', 'TEMPERATUR', 'FREITAG', 'ANFANG', 'SONNTAG'], 'bin_8': ['NEBEL', 'HEISS', 'STURM', 'GEWITTER', 'ORKAN', 'FLUSS', 'NUR', 'MAESSIG', 'MEISTENS', 'ELF', 'LIEB', 'UND', 'LANG', 'LAND', 'BEWOELKT', 'SCHAUER', 'FRISCH', 'MILD', 'ZWANZIG', 'BISSCHEN', 'SCHWACH', 'KUESTE', 'KOENNEN', 'WOCHENENDE', 'ALPEN', 'MOEGLICH'], 'bin_9': ['IM-VERLAUF', 'BERG', 'SECHS', 'SCHON', 'ENORM', 'ACHTZEHN', 'TROCKEN', 'ZEHN', 'GRAD', 'VERSCHWINDEN', 'WOLKE', 'SUED', 'JANUAR', 'EUROPA', 'WARM', 'JETZT', 'NEU', 'OST', 'TEILWEISE', 'poss-EUCH', 'WEST', 'AUCH', 'TIEF', 'WARNUNG', 'NORDWEST', 'REGEN'], 'bin_10': ['MORGEN', 'DEUTSCH', 'FREUNDLICH', 'WIND', 'KOMMEN', 'NORD', 'VIEL', 'HOCH', 'NOCH', 'SONNE', 'GUT', 'SCHOEN', 'MITTE', 'KLAR', 'SCHNEE', 'DANN', 'SONST', 'REGION', 'BESONDERS', 'WETTER', 'WEHEN', 'AB', 'DIENST', 'STEIGEN', 'IX', 'NACHT', 'ABEND', 'MEHR', 'TAG', 'HEUTE']}
#  bin from SignCL
# binned_labels = {'bin_1': ['WO', 'DU', 'negalp-AUCH', 'ICH', 'DAZWISCHEN', 'SELTEN', 'VERAENDERN', 'SECHSHUNDERT', 'UNTER', 'ALLE', 'DURCHGEHEND', 'ZEIGEN-BILDSCHIRM', 'ALLGAEU', 'neg-HABEN', 'GERADE', 'EIN-PAAR-TAGE', 'FEBRUAR', 'KURZ', 'SKANDINAVIEN', 'WIE-IMMER', 'GEFRIEREN', 'SEE', 'METER', 'SINKEN', 'TATSAECHLICH', 'M'], 'bin_2': ['WAS', 'MAI', 'MITTEILEN', 'INFORMIEREN', 'TAGSUEBER', 'ACHTE', 'SCHEINEN', 'HABEN', 'NACHMITTAG', 'DAUER', 'GLEICH', 'MACHEN', 'WINTER', 'ZONE', 'ERSTE', 'VORAUSSAGE', 'VORAUS', 'UNTERSCHIED', 'WENIGER', 'JUNI', 'MAERZ', 'GRAUPEL', 'neg-REGEN', 'SPAETER', 'DARUM', 'SIEBEN'], 'bin_3': ['FUENFTE', 'WIE-AUSSEHEN', 'AUGUST', 'DREI', 'DRITTE', 'DIENSTAG', 'APRIL', 'NULL', 'HOEHE', 'TSCHUESS', 'ZWEI', 'BEWOELKT', 'negalp-KEIN', 'JULI', 'AUFLOESEN', 'WECHSEL', 'NAH', 'SCHOTTLAND', 'DEZEMBER', 'NOVEMBER', 'BURG', 'REST', 'LANGSAM', 'PLUS', 'MITTWOCH', 'BEGRUESSEN'], 'bin_4': ['WUENSCHEN', 'TAUEN', 'WENIG', 'GRUND', 'BODEN', 'WIEDER', 'DABEI', 'OKTOBER', 'ZWISCHEN', 'WEITER', 'MAXIMAL', 'VERSCHIEDEN', 'VORSICHT', 'ABWECHSELN', 'DRUCK', 'IN-KOMMEND', 'WOCHE', 'GLATT', 'HABEN2', 'KOMMEN', 'DREIZEHN', 'EINS', 'BAYERN', 'ZUSCHAUER', 'NAECHSTE', 'HAGEL'], 'bin_5': ['ZUERST', 'HIMMEL', 'VOGEL', 'ACHT', 'SEHEN', 'NEBEL', 'SECHSZEHN', 'VIERZEHN', 'DREISSIG', 'FUENFZEHN', 'WAHRSCHEINLICH', 'BERLIN', 'UND', 'SACHSEN', 'HERBST', 'ANFANG', 'SUEDOST', 'SECHS', 'KOELN', 'DESHALB', 'LOCKER', 'DAZU', 'MEISTENS', 'GRAD', 'SO', 'ABER'], 'bin_6': ['VERSCHWINDEN', 'UEBER', 'ZWOELF', 'LIEB', 'TEIL', 'MITTAG', 'OFT', 'SUEDWEST', 'BRAND', 'WENN', 'WALD', 'KOENNEN', 'DEUTSCHLAND', 'ORT', 'SAMSTAG', 'SEPTEMBER', 'poss-EUCH', 'BLEIBEN', 'STERN', 'SUED', 'BESSER', 'GUT', 'MONTAG', 'VIER', 'ORKAN', 'NACH'], 'bin_7': ['FUENF', 'TRUEB', 'ZWEITE', 'FLUSS', 'NEUNZEHN', 'BISSCHEN', 'HAUPTSAECHLICH', 'WIE', 'MINUS', 'RUHIG', 'MISCHUNG', 'SONNTAG', 'ZEHN', 'TEMPERATUR', 'LAND', 'JETZT', 'SCHNEIEN', 'UEBERWIEGEND', 'LANG', 'NORDOST', 'KALT', 'ACHTZEHN', 'JANUAR', 'HOCH', 'KUEHL', 'SIEBZEHN'], 'bin_8': ['ALPEN', 'ZWANZIG', 'NUR', 'MIT', 'MAL', 'WECHSELHAFT', 'SCHWER', 'BERG', 'SCHON', 'MORGEN', 'AEHNLICH', 'LEICHT', 'MOEGLICH', 'STRASSE', 'WEHEN', 'WOLKE', 'BIS', 'STURM', 'FRISCH', 'ELF', 'GEFAHR', 'LUFT', 'WOCHENENDE', 'REGION', 'REGEN', 'SONST'], 'bin_9': ['OST', 'VOR', 'FRUEH', 'MILD', 'TIEF', 'FROST', 'EUROPA', 'KUESTE', 'SONNE', 'NORDWEST', 'DEUTSCH', 'STARK', 'GEWITTER', 'FREITAG', 'SCHWACH', 'DOCH', 'WEST', 'SCHAUER', 'DONNERSTAG', 'MEER', 'FEUCHT', 'BESONDERS', 'SPEZIELL', 'MITTE', 'NORD', 'UNWETTER'], 'bin_10': ['AB', 'HEISS', 'AUCH', 'WIND', 'WARM', 'IM-VERLAUF', 'NEUN', 'IX', 'KLAR', 'NEU', 'WETTER', 'WARNUNG', 'ODER', 'VIEL', 'TAG', 'ENORM', 'SCHNEE', 'MEHR', 'MAESSIG', 'HEUTE', 'DANN', 'TROCKEN', 'NACHT', 'ABEND', 'SCHOEN', 'NOCH', 'DIENST', 'FREUNDLICH', 'TEILWEISE', 'STEIGEN']}

# bin from CLIP 
# binned_labels = {'bin_1': ['WO', 'DU', 'SELTEN', 'ICH', 'DAZWISCHEN', 'ALLE', 'UNTER', 'ALLGAEU', 'SECHSHUNDERT', 'VERAENDERN', 'KURZ', 'neg-HABEN', 'WINTER', 'negalp-AUCH', 'ZEIGEN-BILDSCHIRM', 'SEE', 'DURCHGEHEND', 'EIN-PAAR-TAGE', 'GEFRIEREN', 'TATSAECHLICH', 'MACHEN', 'VORAUSSAGE', 'MITTEILEN', 'M', 'UNTERSCHIED', 'ACHTE'], 'bin_2': ['VORSICHT', 'ERSTE', 'DRITTE', 'SKANDINAVIEN', 'ZONE', 'NULL', 'DAUER', 'WAS', 'GERADE', 'METER', 'GLEICH', 'WENIGER', 'HABEN', 'TAGSUEBER', 'WIE-IMMER', 'DARUM', 'GRAUPEL', 'neg-REGEN', 'NOVEMBER', 'SIEBEN', 'FUENFTE', 'ABWECHSELN', 'SCHOTTLAND', 'SINKEN', 'WIEDER', 'NACHMITTAG'], 'bin_3': ['SCHEINEN', 'WECHSEL', 'DREI', 'ZWEI', 'VORAUS', 'DIENSTAG', 'NAH', 'INFORMIEREN', 'TSCHUESS', 'JUNI', 'MITTWOCH', 'MAI', 'HOEHE', 'HABEN2', 'KOELN', 'GRUND', 'WIE-AUSSEHEN', 'REST', 'VOGEL', 'PLUS', 'ZWEITE', 'VIERZEHN', 'HAGEL', 'AUFLOESEN', 'GLATT', 'BURG'], 'bin_4': ['BAYERN', 'NAECHSTE', 'APRIL', 'FUENFZEHN', 'SECHSZEHN', 'TAUEN', 'DREISSIG', 'ZUERST', 'WOCHE', 'LOCKER', 'OFT', 'FUENF', 'LANGSAM', 'negalp-KEIN', 'ZWISCHEN', 'WEITER', 'WENIG', 'DREIZEHN', 'EINS', 'UEBER', 'DRUCK', 'ACHT', 'BRAND', 'SCHWER', 'IN-KOMMEND', 'MAXIMAL'], 'bin_5': ['MONTAG', 'WENN', 'SPAETER', 'HIMMEL', 'DOCH', 'SACHSEN', 'FEBRUAR', 'SUEDOST', 'DABEI', 'SUEDWEST', 'HERBST', 'STRASSE', 'SONNTAG', 'ZWOELF', 'SECHS', 'TRUEB', 'WALD', 'UND', 'ORT', 'SIEBZEHN', 'VERSCHIEDEN', 'BERLIN', 'NEUNZEHN', 'WUENSCHEN', 'DEUTSCHLAND', 'VOR'], 'bin_6': ['SAMSTAG', 'BODEN', 'LEICHT', 'AUGUST', 'MITTAG', 'ACHTZEHN', 'ZWANZIG', 'TEMPERATUR', 'AEHNLICH', 'TEIL', 'MAERZ', 'poss-EUCH', 'ZUSCHAUER', 'DAZU', 'BEWOELKT', 'ELF', 'FLUSS', 'BLEIBEN', 'FRUEH', 'SEPTEMBER', 'NEBEL', 'LAND', 'WIE', 'ANFANG', 'WAHRSCHEINLICH', 'NEUN'], 'bin_7': ['KALT', 'UNWETTER', 'EUROPA', 'BESSER', 'HAUPTSAECHLICH', 'SCHNEIEN', 'MIT', 'DESHALB', 'SUED', 'OKTOBER', 'MISCHUNG', 'STERN', 'UEBERWIEGEND', 'DEZEMBER', 'SPEZIELL', 'MAL', 'MINUS', 'VIER', 'SO', 'GRAD', 'ABER', 'DONNERSTAG', 'BEGRUESSEN', 'RUHIG', 'BIS', 'WECHSELHAFT'], 'bin_8': ['SEHEN', 'HOCH', 'STURM', 'LIEB', 'ZEHN', 'KUEHL', 'NUR', 'LUFT', 'FEUCHT', 'LANG', 'HEISS', 'MEISTENS', 'KOENNEN', 'WOCHENENDE', 'AB', 'GEFAHR', 'KOMMEN', 'SONST', 'NACH', 'IM-VERLAUF', 'MEER', 'GEWITTER', 'ODER', 'FRISCH', 'FREITAG', 'VERSCHWINDEN'], 'bin_9': ['TIEF', 'STARK', 'SCHWACH', 'GUT', 'WEST', 'NORDOST', 'ENORM', 'NEU', 'SCHAUER', 'OST', 'MORGEN', 'REGEN', 'BISSCHEN', 'ALPEN', 'JANUAR', 'VIEL', 'MILD', 'NORDWEST', 'BERG', 'DEUTSCH', 'FROST', 'MAESSIG', 'WEHEN', 'SCHON', 'AUCH', 'WARM'], 'bin_10': ['KUESTE', 'ORKAN', 'REGION', 'JETZT', 'MOEGLICH', 'WIND', 'WOLKE', 'WETTER', 'SCHNEE', 'SONNE', 'BESONDERS', 'NORD', 'WARNUNG', 'TEILWEISE', 'TAG', 'MITTE', 'KLAR', 'NOCH', 'IX', 'NACHT', 'JULI', 'STEIGEN', 'HEUTE', 'TROCKEN', 'FREUNDLICH', 'MEHR', 'SCHOEN', 'DIENST', 'ABEND', 'DANN']}

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Gloss-free Sign Language Translation script", add_help=False
    )
    parser.add_argument('--debug', action="store_true", help="disable distributed setting")
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
    # parser.add_argument("--finetune", action="store_true", help="load pretrained visual and text encoder for SLT finetuning")
    
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
    parser.add_argument("--ouput_feature", action="store_true")
    # parser.add_argument("--bin_bleu", action="store_true")

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

    dev_data = S2T_Dataset(
        path=config["data"]["dev_label_path"],
        tokenizer=tokenizer,
        config=config,
        args=args,
        phase="dev",
        reg_label_path=config["data"]["dev_reg_label_path"]
    )
    print("dev dataset", dev_data)
    if args.debug:
        dev_sampler = None
    else:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data, shuffle=False)
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
    if args.debug:
        test_sampler = None
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
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

    # test_stats = evaluate(
    #     args,
    #     dev_dataloader,
    #     model,
    #     model_without_ddp,
    #     tokenizer,
    #     criterion,
    #     config,
    #     UNK_IDX,
    #     SPECIAL_SYMBOLS,
    #     PAD_IDX,
    #     device,
    # )
    # print(
    #     f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} "
    # )
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
            pred_by_recognition_label = {}
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
                for idx, l in enumerate(src_input["recognition_label"]):
                    for l2 in set(l):
                        # tgt_pres.append(output[i, :])
                        # tgt_refs.append(tgt_input["input_ids"][i, :])
                        if l2 == "<PAD>":
                            continue
                        elif l2 not in pred_by_recognition_label:
                            pred_by_recognition_label[l2] = [[output[idx, :]], [tgt_input["input_ids"][idx, :]]]
                        else:
                            pred_by_recognition_label[l2][0].append(output[idx, :])
                            pred_by_recognition_label[l2][1].append(tgt_input["input_ids"][idx, :])
                # break
        if args.ouput_feature:
            # utils.tsne_visualize(frame_feats_by_recognition_label, save_dir=args.output_dir)
            # utils.tsne_visualize_topk(frame_feats_by_recognition_label, save_dir=args.output_dir)
            import pickle
            base_filename = "frame_features"
            extension = ".pickle"
            i = 1
            while os.path.exists(os.path.join(args.output_dir, f"{base_filename}_{i}{extension}")):
                i += 1
            save_path = os.path.join(args.output_dir, f"{base_filename}_{i}{extension}")
            with open(save_path, 'wb') as f:
                pickle.dump(frame_feats_by_recognition_label, f)
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

    bin_bleu_scores = {}
    for bin_label, labels in binned_labels.items():
        bin_preds = []
        bin_refs = []
        for label in labels:
            if label in pred_by_recognition_label:
                preds, refs = pred_by_recognition_label[label]
                bin_preds.extend(preds)
                bin_refs.extend(refs)
        if bin_preds and bin_refs:
            bin_preds = tokenizer.batch_decode(bin_preds, skip_special_tokens=True)
            bin_refs = tokenizer.batch_decode(bin_refs, skip_special_tokens=True)
            bin_bleu_score = bleu.corpus_score(bin_preds, [bin_refs]).score
            
            bin_bleu_scores[bin_label] = bin_bleu_score
            print(f"BLEU-4 score for {bin_label}: {bin_bleu_score:.2f}")

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
