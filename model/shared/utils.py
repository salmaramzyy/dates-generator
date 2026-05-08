import torch
from datetime import datetime, date
from typing import List, Tuple
import matplotlib.pyplot as plt

from tokenizer import MIN_DECADE

DAY_MAP   = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
MONTH_MAP = {0:1,1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10,10:11,11:12}


def parse_generated_date(date_str: str) -> date | None:
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").date()
    except ValueError:
        return None


def check_conditions(date_str: str, day_idx: int, month_idx: int,
                     leap_idx: int, decade_idx: int) -> Tuple[bool,bool,bool,bool,bool]:
    d = parse_generated_date(date_str)
    if d is None:
        return False, False, False, False, False

    day_ok   = (d.weekday() == day_idx)
    month_ok = (d.month == MONTH_MAP[month_idx])

    year     = d.year
    is_leap  = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    leap_ok  = (is_leap == bool(leap_idx))

    decade_token = decade_idx + MIN_DECADE
    decade_ok    = (year // 10 == decade_token)

    return True, day_ok, month_ok, leap_ok, decade_ok


def condition_accuracy(predictions: List[str],
                       conditions: List[Tuple[int,int,int,int]]) -> dict:
    total = len(predictions)
    valid = day = month = leap = decade = all_pass = 0

    for pred, (d_idx, m_idx, l_idx, dec_idx) in zip(predictions, conditions):
        v, d_ok, m_ok, l_ok, dec_ok = check_conditions(pred, d_idx, m_idx, l_idx, dec_idx)
        if v:                              valid    += 1
        if d_ok:                           day      += 1
        if m_ok:                           month    += 1
        if l_ok:                           leap     += 1
        if dec_ok:                         decade   += 1
        if v and d_ok and m_ok and l_ok and dec_ok: all_pass += 1

    return {
        "valid_date" : valid    / total * 100,
        "day_acc"    : day      / total * 100,
        "month_acc"  : month    / total * 100,
        "leap_acc"   : leap     / total * 100,
        "decade_acc" : decade   / total * 100,
        "all_pass"   : all_pass / total * 100,
    }


def plot_losses(train_losses: List[float], val_losses: List[float],
                title: str, save_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics: dict, save_path: str) -> None:
    with open(save_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.2f}%\n")