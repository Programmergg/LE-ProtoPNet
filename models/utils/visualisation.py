import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Adapted from
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py#L33
def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = True, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    This function overlays the cam mask on the image as an heatmap. By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
   # heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 *  mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    if np.max(img) > 1.05:  # sometimes there are small numerical errors
        raise Exception("The input image should np.float32 in the range [0, 1]")
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def plot_similarity(samples: torch.tensor, pred_dict: dict, importance: torch.tensor,preds, mean_normalize: torch.tensor, std_normalize: torch.tensor, fig_nb: int, nb_proto_plot=5, nb_pred=0):
    unnormalize = transforms.Normalize((-mean_normalize / std_normalize), (1.0 / std_normalize))
    fig, axs = plt.subplots(ncols=nb_proto_plot + 1, nrows=fig_nb, figsize=(3 * nb_proto_plot, 3 * fig_nb))
    for idx_image in range(fig_nb):
        importance_plotted = 0
        img_size = samples[idx_image].shape[1]
        image = unnormalize(samples[idx_image]).permute(1, 2, 0).detach().cpu().numpy()
        #image = samples[idx_image].permute(1, 2, 0).detach().cpu().numpy()
        sorted_class = torch.argsort(preds[idx_image], descending=True)
        #sorted_class = torch.argsort(pred_dict["pred"][idx_image], descending=True)
        #importance = pred_dict["importance"][idx_image, :, sorted_class[nb_pred]]
       # importance_sorted = importance[idx_image, :, sorted_class[nb_pred]]
        importance_sorted = importance[:, sorted_class[nb_pred]]
        idx_mask_max = torch.argsort(importance_sorted, descending=True)[:nb_proto_plot]
        axs[idx_image, 0].imshow(image)
        for i in range(nb_proto_plot):
            idx_mask_tmp = idx_mask_max[i]
            similiarity_tmp = pred_dict["similarity_prototype"][idx_image, idx_mask_tmp]
            size_square_similarity = int(math.sqrt(similiarity_tmp.shape[0]))
            similiarity_tmp = similiarity_tmp.reshape(size_square_similarity, size_square_similarity)
            similarity_scaled = torch.nn.functional.interpolate(similiarity_tmp[None, None, :, :], size=(img_size, img_size), scale_factor=None, mode="bilinear")
            similarity_plot = show_cam_on_image(image, similarity_scaled[0, 0].detach().cpu().numpy(),)
            axs[idx_image, i + 1].imshow(similarity_plot)
            axs[idx_image, i + 1].title.set_text(f"{idx_mask_tmp}; importance: {importance_sorted[idx_mask_tmp].detach().cpu().numpy():.2f}",)
            importance_plotted += importance_sorted[idx_mask_tmp].detach().cpu()
        
        axs[idx_image, 0].title.set_text(f"Pred. class: {sorted_class[0]} \n score: {preds[idx_image, sorted_class[0]]:.1f} ({(importance_plotted/importance_sorted.sum())*100:.1f}%)")
        #axs[idx_image, 0].title.set_text(f"Pred. class: {sorted_class[0]} \n score: {pred_dict['pred'][idx_image, sorted_class[0]]:.1f} ({(importance_plotted/importance.sum())*100:.1f}%)")
        plt.tight_layout()
       # break
    return fig

def plot_weight_heatmap(weights):
    # Assuming 'classification_head' is your model
    fig, ax = plt.subplots(figsize=(10, 10))
    # Create a heatmap of the weights
    cax = ax.imshow(weights, cmap="hot")
    # Add a colorbar to the figure
    fig.colorbar(cax)
    # Set the title of the plot
    ax.set_title("Weight Heatmap")
    # Return the figure
    return fig


def plot_weight_distribution(weights):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(weights.flatten(), bins=30)
    ax.set_title("Weight Distribution")
    return fig