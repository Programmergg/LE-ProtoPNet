import torch
import numpy as np
from torch import nn
from tqdm import tqdm

def get_part_importance(image, part_map, colors_to_part):
    """
    Outputs part importances for each part.
    """
    assert image.shape[0] == 1  # B = 1
    part_importances = {}
    dilation1 = nn.MaxPool2d(5, stride=1, padding=2)
    for part_color in colors_to_part.keys():
        torch_color = torch.zeros(1, 3, 1, 1).to(image.device)
        torch_color[0, 0, 0, 0] = part_color[0]
        torch_color[0, 1, 0, 0] = part_color[1]
        torch_color[0, 2, 0, 0] = part_color[2]
        color_available = torch.all(part_map == torch_color, dim=1, keepdim=True).float()
        attribution_part = dilation1(color_available)
        part_string = colors_to_part[part_color]
        part_string = "".join((x for x in part_string if x.isalpha()))
        part_importances[part_string] = attribution_part
    return part_importances

def perturb_img(norm_img, std=0.2, eps=0.25):
    noise = torch.zeros(norm_img.shape).normal_(mean=0, std=std).cuda()
    noise = torch.clip(noise, min=-eps, max=eps)  # Constrain the maximum absolute value, ensuring that the noise is imperceptible by humans
    perturb_img = norm_img + noise
    return perturb_img

def compute_consistency(all_proto_part_mask, all_part_presence, target_all, threshold=0.8):
    dict_consistency = {}
    all_score = []
    for target in np.unique(target_all):
        idx_same_target = np.argwhere(np.array(target_all) == target).squeeze()
        all_proto_to_part_tmp = all_proto_part_mask[idx_same_target]
        all_proto_to_part_tmp = all_proto_to_part_tmp > 0.0
        all_proto_part_mask_tmp = all_part_presence[idx_same_target]
        consistency = all_proto_to_part_tmp.sum(axis=0) / all_proto_part_mask_tmp.sum(axis=0)
        consistency = consistency[consistency.max(axis=1) > 0]
        score = consistency.max(axis=1) > threshold
        normalised_score = score.sum() / len(score)
        all_score.append(score)
        dict_consistency[target] = normalised_score
    mean_score = np.concatenate(all_score).sum() / len(np.concatenate(all_score))
    return mean_score

def attribute_analysis(model, test_dataset, test_loader, add_noise=False):
    list_image = []
    output_all = []
    list_all_part_proto, list_all_presence, target_all = [], [], []
    colors_to_part = test_dataset.colors_to_part
    bird_parts_keys = list(test_dataset.parts.keys())
    for sample in tqdm(test_loader):
        image = sample["image"]
        target = sample["class_idx"]
        part_map = sample["part_map"].cuda()
        image = image.cuda()
        if add_noise:
            image = perturb_img(image)
        list_image.append(image)
        with torch.no_grad():
            output = model(image)
        output_all.append(output)
        target_all.append(target.detach().cpu().item())
        importance = output["importance"][0][:, target]
        importance[importance < 0.1] = 0
        img_size = image.shape[2:]
        similarity = output["similarity_prototype"][0]
        # similarity[similarity < 0.1] = 0
        similarity_weighted = output["similarity_prototype"][0] * importance
        h = w = int(similarity.shape[-1] ** 0.5)
        similarity_weighted = similarity_weighted.reshape(similarity.shape[0], h, w)
        similarity_weighted = similarity_weighted.unsqueeze(1)
        print("similarity_weighted", similarity_weighted.shape)
        similarity_reshaped = torch.nn.functional.interpolate(similarity_weighted, size=(img_size), scale_factor=None)
        similarity_reshaped = similarity_reshaped.squeeze(1)
        part_num = len(bird_parts_keys)
        part_presence = np.zeros(part_num)
        np_part_proto = np.zeros((similarity_reshaped.shape[0], int(part_num)))
        part_importances = get_part_importance(image, part_map, colors_to_part)
        for part_id, remove_part in enumerate(bird_parts_keys):
            importance_map = part_importances[remove_part]
            part_presence[part_id] = importance_map.sum() > 0
            importance_part = similarity_reshaped * importance_map[0]
            np_part_proto[:, part_id] = (importance_part.amax(dim=(1, 2)).cpu().numpy() > 0.2)
        list_all_part_proto.append(np_part_proto)
        list_all_presence.append(part_presence)
    return list_all_part_proto, list_all_presence, target_all