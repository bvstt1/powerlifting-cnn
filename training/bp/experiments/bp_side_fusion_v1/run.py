"""python -m training.bp.experiments.bp_side_fusion_v1.run --view left|right --phase prepare|train|all

No lee Test. Ambos pipelines se ejecutan de manera independiente por vista.
"""
import argparse

from .data import prepare
from .train import CONFIG, train_view


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--view",required=True,choices=("left","right"))
    p.add_argument("--phase",required=True,choices=("prepare","train","all"))
    args=p.parse_args()
    if args.phase in ("prepare","all"):prepare(args.view,CONFIG)
    if args.phase in ("train","all"):train_view(args.view)


if __name__=="__main__":main()
