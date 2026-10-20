"""Uso: python -m training.bp.experiments.bp_left_04_cv_v1.run --phase prepare|train|all|report|verify."""
import argparse

from training.bp.experiments.bp_side_cv_v1 import report, summarize, train, verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",required=True,choices=("prepare","train","all","report","verify"))
    phase=p.parse_args().phase
    if phase in ("prepare","all"):
        train.prepare("left")
        train.save_json(train.output("left")/"cv_config.json",train.CONFIG)
    if phase in ("train","all"):train.train("left")
    if phase in ("report","all"):
        summarize.analyze("left");report.detail("left")
    if phase=="verify":verify.run("left")


if __name__=="__main__":main()
