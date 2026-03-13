# Adapted from funny birds: https://github.com/visinf/funnybirds-framework?tab=readme-ov-file
import torch
import numpy as np
import torch.nn as nn
from abc import abstractmethod

class AbstractExplainer:
    def __init__(self, model, baseline=None):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model
        self.explainer_name = type(self.model).__name__
        self.baseline = baseline
        print(self.explainer_name)

    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

class AbstractAttributionExplainer(AbstractExplainer):
    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)

    # image: the input image
    # part map: the corresponding segmentation map where one color denotes one part
    # target: the target class
    # colors_to_part: a list that maps colors to parts
    # thresholds: the different thresholds to use to estimate which parts are important
    # with_bg: include the background parts in the computation
    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg=False):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1  # B = 1
        attribution = self.explain(image, target=target)
        part_importances = self.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)
        important_parts_for_thresholds = []
        for threshold in thresholds:
            important_parts = []
            for key in part_importances.keys():
                if part_importances[key] > (attribution.sum() * threshold):
                    important_parts.append(key)
            important_parts_for_thresholds.append(important_parts)
        return important_parts_for_thresholds

    # image: the input image
    # part map: the corresponding segmentation map where one color denotes one part
    # target: the target class
    # colors_to_part: a list that maps colors to parts
    # with_bg: include the background parts in the computation
    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg=False):
        """
        Outputs part importances for each part.
        """
        assert image.shape[0] == 1  # B = 1
        attribution = self.explain(image, target=target)
        part_importances = {}
        dilation1 = nn.MaxPool2d(5, stride=1, padding=2)
        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1, 3, 1, 1).to(image.device)
            torch_color[0, 0, 0, 0] = part_color[0]
            torch_color[0, 1, 0, 0] = part_color[1]
            torch_color[0, 2, 0, 0] = part_color[2]
            color_available = torch.all(part_map == torch_color, dim=1, keepdim=True).float()
            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            attribution_in_part = attribution_in_part.sum()
            part_string = colors_to_part[part_color]
            part_string = "".join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()
        if with_bg:
            for i in range(50):
                torch_color = torch.zeros(1, 3, 1, 1).to(image.device)
                torch_color[0, 0, 0, 0] = 204
                torch_color[0, 1, 0, 0] = 204
                torch_color[0, 2, 0, 0] = 204 + i
                color_available = torch.all(part_map == torch_color, dim=1, keepdim=True).float()
                color_available_dilated = dilation1(color_available)
                attribution_in_part = attribution * color_available_dilated
                attribution_in_part = attribution_in_part.sum()
                bg_string = "bg_" + str(i).zfill(3)
                part_importances[bg_string] = attribution_in_part.item()
        return part_importances

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

class ProtoExplainer(AbstractAttributionExplainer):
    def explain(self, input, target):
        pred_dict = self.model(input)
        img_size = input.shape[2:]
        similarity = pred_dict["similarity_prototype"][0] 
        h = w = int(similarity.shape[-1] ** 0.5)
        similarity = similarity.reshape(similarity.shape[0], h, w)
        similarity = similarity.unsqueeze(1)
        similarity_scaled = torch.nn.functional.interpolate(similarity, size=(img_size), scale_factor=None)
        similarity_scaled = similarity_scaled.squeeze(1).reshape(similarity.shape[0], -1)
        similarity_scaled = similarity_scaled / similarity_scaled.sum(dim=1)[:, None]
        similarity_weighted = torch.einsum(
            "pi,bp->bi",
            similarity_scaled,
            pred_dict["importance"][..., target].squeeze(-1),
        )
        similarity_weighted = similarity_weighted[0].reshape(img_size, img_size)
        return similarity_weighted