import os
import random
import argparse
import pyrootutils

import torch
import torch.nn as nn

pyrootutils.setup_root(os.getcwd(), indicator="requirements.txt", pythonpath=True)

from evaluation.funny_birds.evaluation_protocols import (
    accuracy_protocol,
    background_independence_protocol,
    controlled_synthetic_data_check_protocol,
    deletion_check_protocol,
    distractibility_protocol,
    preservation_check_protocol,
    single_deletion_protocol,
    target_sensitivity_protocol,
)
from shared_utils.utils_experiments import load_model_dataset
from evaluation.funny_birds.plot_results import plot_results_funny_birds
from evaluation.funny_birds.explainers.explainer_wrapper import ProtoExplainer

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class ModelExplainerWrapper(nn.Module):
    def __init__(self, model, explainer=None,):
        """
        A generic wrapper that takes any model and any explainer to putput model predictions
        and explanations that highlight important input image part.
        Args:
            model: PyTorch neural network model
            explainer: PyTorch model explainer
        """
        super().__init__()
        self.model = model
        self.explainer = explainer

    def forward(self, input):
        output = self.model.forward(input)
        output = output["pred"]
        return output

    def predict(self, input):
        output = self.model.forward(input)
        breakpoint()

    def explain(self, input):
        return self.explainer.explain(self.model, input)

def main(path_sim):
    random.seed(42)
    torch.manual_seed(42)
    model, dataloader = load_model_dataset(path_sim, "test")
    explainer = ProtoExplainer(model)
    model = ModelExplainerWrapper(model)
    model.cuda()
    data_path = "/workspaces/intrainpretable/data/FunnyBirds"
    batch_size = 64
    model.eval()
    print("Computing accuracy...")
    accuracy = accuracy_protocol(model, data_path, batch_size)
    accuracy = round(accuracy, 5)
    print("Computing controlled synthetic data check...")
    csdc = controlled_synthetic_data_check_protocol(model, explainer, data_path)
    print("Computing target sensitivity...")
    ts = target_sensitivity_protocol(model, explainer, data_path)
    ts = round(ts, 5)
    print("Computing single deletion...")
    sd = single_deletion_protocol(model, explainer, data_path)
    sd = round(sd, 5)
    print("Computing preservation check...")
    pc = preservation_check_protocol(model, explainer, data_path)
    print("Computing deletion check...")
    dc = deletion_check_protocol(model, explainer, data_path)
    print("Computing distractibility...")
    distractibility = distractibility_protocol(model, explainer, data_path)
    print("Computing background independence...")
    background_independence = background_independence_protocol(model, data_path)
    background_independence = round(background_independence, 5)

    # select completeness and distractability thresholds such that they maximize the sum of both
    max_score = 0
    best_threshold = -1
    for threshold in csdc.keys():
        max_score_tmp = (csdc[threshold] / 3.0 + pc[threshold] / 3.0 + dc[threshold] / 3.0 + distractibility[threshold])
        if max_score_tmp > max_score:
            max_score = max_score_tmp
            best_threshold = threshold

    print("FINAL RESULTS:")
    print("Accuracy, CSDC, PC, DC, Distractability, Background independence, SD, TS")
    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(accuracy, round(csdc[best_threshold], 5), round(pc[best_threshold], 5), round(dc[best_threshold], 5),
                                                  round(distractibility[best_threshold], 5), background_independence, sd, ts))
    print("Best threshold:", best_threshold)
    results = [accuracy, round(csdc[best_threshold], 5), round(pc[best_threshold], 5), round(dc[best_threshold], 5), round(distractibility[best_threshold], 5), background_independence, sd, ts]
    plot_results_funny_birds(results, path_sim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FunnyBirds - Explanation Evaluation")
    args = parser.parse_args()
    parser.add_argument("--path_sim", metavar="DIR", required=True, help="path to datset with model and config")
    main(args.path_sim)