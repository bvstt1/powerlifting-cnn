"""Uso: python -m training.bp_right_fusion_v1.run --phase prepare|train|all."""
import argparse
from training.bp_side_fusion_v1.data import prepare
from training.bp_side_fusion_v1.train import CONFIG, train_view


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",required=True,choices=("prepare","train","all"))
    phase=p.parse_args().phase
    if phase in ("prepare","all"):prepare("right",CONFIG)
    if phase in ("train","all"):train_view("right")


if __name__=="__main__":main()
