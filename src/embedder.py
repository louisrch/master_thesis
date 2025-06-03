import clip
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import os
import utils

# Predefined mean and std values for normalization
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class Embedder:
    """
    Embedding model used to project pictures into a latent space
    useful arguments in kwargs:
    model : str -> name of the model
    dvc : str -> device on which to compute the embedding
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.model == "CLIP":
            model, _ = clip.load("ViT-B/32", device=self.dvc)
            self.encoder = model
            self.std = torch.tensor(CLIP_STD, device=self.dvc).view(1, 3, 1, 1)
            self.mean = torch.tensor(CLIP_MEAN, device=self.dvc).view(1, 3, 1, 1)
        elif self.model == "DINOV2":
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14_reg'
            ).to(self.dvc)
            self.std = torch.tensor(DINO_STD, device=self.dvc).view(1, 3, 1, 1)
            self.mean = torch.tensor(DINO_MEAN, device=self.dvc).view(1, 3, 1, 1)
        self.img_size = 224

    def get_fast_preprocessing_numpy(
        self,
        img_np: np.ndarray,
        output_height: int = 224,
        output_width: int = 224
    ) -> torch.Tensor:
        """
        img_np : N x H x W x 3 RGB image np ndarray
        output : N x 3 x output_height x output_width image
        """
        img_tensor = torch.from_numpy(img_np)
        # if we only have one image (HxWx3 case), we add another dimension
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # permute the axes such that we obtain the desired N x 3 x H x W shape
        img_tensor = (
            img_tensor.permute(0, 3, 1, 2)
            .to(self.dvc, non_blocking=True)
            .float()
            .div_(255.)
        )

        # downsample the image
        img_tensor = F.interpolate(
            img_tensor,
            size=(output_height, output_width),
            mode='bilinear',
            align_corners=False
        )

        # normalize the image to the model's mean and std 
        img_tensor = (img_tensor - self.mean) / self.std
        return img_tensor

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        image : torch.Tensor - input image, assumed to be preprocessed
        returns the embedding of the image
        """
        assert image.shape[2] == self.img_size and image.shape[3] == self.img_size
        if self.model == "CLIP":
            with torch.no_grad():
                return self.encoder.encode_image(image)
        elif self.model == "DINOV2":
            with torch.no_grad():
                return self.encoder.forward(image)

    def get_state_embedding(self, env: gym.Env) -> torch.Tensor:
        """
        takes a snapshot of the environment, i.e. the depiction of the current state,
        and forwards it to the embedding model
        env : RL environment
        """
        rendered = env.render()
        return self.get_image_embedding(rendered)

    def get_image_embedding(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocesses the image and forwards it into the embedding model.
        Returns the projection of the image onto the latent space of the embedding model.
        """
        preprocessed = self.get_fast_preprocessing_numpy(image)
        return self.forward_image(preprocessed)


