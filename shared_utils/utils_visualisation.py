import cv2
import torch
import numpy as np
import seaborn as sns
from os.path import join as pj
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def return_colorblind_palette() -> np.ndarray:
    """
    Returns a colorblind palette of 12 colors.
    Returns:
        np.ndarray: A colorblind palette of 12 colors.
    """
    colorblind_palette = [
        (166, 206, 227),
        (31, 120, 180),
        (178, 223, 138),
        (51, 160, 44),
        (251, 154, 153),
        (227, 26, 28),
        (253, 191, 111),
        (255, 127, 0),
        (202, 178, 214),
        (106, 61, 154),
        (255, 255, 153),
        (177, 89, 40),
    ]
    colorblind_palette = np.array(colorblind_palette) / 255
    return colorblind_palette

def plot_prototypes(image, similarity_image, alpha=0.5, color_annotations=None, axs=None, pred=None, label=None,):
    # extract the top proto_per_figure from importance
    if axs is None:
        fig, axs = plt.subplots(1, figsize=(10, 10))
    else:
        fig = None
    if color_annotations is None:
        color_annotations = sns.color_palette("hls", len(similarity_image))
    array_masks = []
    array_edges = []
    img_size = image.shape[1:]
    size_square_similarity = int(similarity_image.shape[1] ** 0.5)
    for idx in range(len(similarity_image)):
        color_annotation = color_annotations[idx]
        similarity_proto = similarity_image[idx].reshape(size_square_similarity, size_square_similarity)
        similarity_proto[similarity_proto < similarity_proto.max() * 0.2] = 0
        # similarity_proto[similarity_proto<0.1] = 0
        similarity_scaled = torch.nn.functional.interpolate(torch.tensor(similarity_proto[None, None, :, :]), size=img_size)
        similarity_scaled = similarity_scaled.detach().cpu().numpy()[0, 0]
        image2 = np.zeros_like(similarity_scaled)
        image2[similarity_scaled > 0] = 100
        edges = cv2.Canny(np.uint8(image2), threshold1=100, threshold2=100)
        edges = cv2.dilate(edges, (3, 3), iterations=1)
        normalized_fill = np.repeat(image2[:, :, None], 3, axis=-1) / max(np.max(image2), 1)
        normalized_edges = np.transpose(edges[None,], (1, 2, 0)) / max(np.max(edges), 1)
        normalized_edges = np.repeat(normalized_edges, 3, axis=-1)
        mask = (normalized_fill * np.array(color_annotation)[:3]).astype(np.float32)
        edges = normalized_edges * np.array(color_annotation)[:3]
        array_masks.append(mask)
        array_edges.append(edges)
    np_mask = np.stack(array_masks)
    np_edges = np.stack(array_edges)
    binary_mask = np.where(np_mask.sum(axis=-1) > 0, 1, 0)
    binary_mask = np.sum(binary_mask, axis=0)
    binary_mask = np.where(binary_mask > 0, 1 / binary_mask, 0)
    alpha_mask = binary_mask
    binary_edge = np.where(np_edges.sum(axis=-1) > 0, 1, 0)
    binary_edge = np.sum(binary_edge, axis=0)
    binary_edge = np.where(binary_edge > 0, 1 / binary_edge, 0)
    alpha_edge = binary_edge
    mask_edge = np.where(np_edges.sum(axis=0) > 0, 0, 1)
    annotation_mask = (np_mask.sum(axis=0) * alpha_mask[:, :, None] * alpha * mask_edge + np_edges.sum(axis=0) * alpha_edge[:, :, None])
    # annotation_mask = np_edges.sum(axis=0)*alpha_edge[:,:,None]
    alpha_image = np.where(annotation_mask.sum(axis=-1) > 0, 1, 1)
    image_plot = (np.transpose(image, (1, 2, 0)) * (1 - alpha) * mask_edge * alpha_image[:, :, None] + annotation_mask)
    if axs is None:
        axs.imshow(image_plot)
        axs.axis("off")
    else:
        axs.imshow(image_plot)
        if (pred is not None) & (label is not None):
            title = f"Class {label} - Pred {pred}"
            # axs.set_title(title)
            # Create a rectangle patch for the border
            if pred == label:
                border = Rectangle((0, 0), image_plot.shape[1], image_plot.shape[0], linewidth=4, edgecolor="lime", facecolor="none")
            else:
                border = Rectangle((0, 0), image_plot.shape[1], image_plot.shape[0], linewidth=4, edgecolor="r", facecolor="none")
        # Add the border to the image
        axs.add_patch(border)
        axs.axis("off")
        return axs

# Adapted from
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/utils/image.py#L33
def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = True, alpha: float = 1, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    # scale mask value between 0 and 1
    if np.max(mask) < 0.2:
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
    heatmap = cv2.applyColorMap(np.uint8(255 * (mask)), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = (np.float32(heatmap) / 255) * alpha
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)