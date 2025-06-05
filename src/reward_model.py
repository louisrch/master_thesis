from embedder import Embedder
import torch
import utils
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F

FLAG_DICT = {"visual" : "v",
            "semantic" : "s",
            "oracle" : "oracle"}

ASSETS_PATH = "assets/"


class RewardModel:
    """
    Reward Model that computes rewards with the help of an embedding model.
    """
    embedding_model : Embedder
    goals : torch.tensor
    def __init__(self, embedding_model: Embedder, goal_path, **kwargs):
        """
        embedding_model : Embedding model used to project states onto a latent space
        goals : dictionary of the respective embedding of each goal image
        maps environment names to depictions
        """
        self.__dict__.update(kwargs)
        self.embedding_model = embedding_model
        self.goals = self.build_goal_list(goal_path)
        self.index = 0

    def compute_rewards(
        self,
        images: list,
        goal_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the associated rewards to each image with respect to the given embedding of the goal state
        """
        images_np = np.stack(images, axis=0)
        image_embeddings = self.embedding_model.get_image_embedding(images_np)
        distance = self.compute_distance(image_embeddings, goal_embedding)
        r = torch.exp(-distance)
        return r

    def compute_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """
        computes the distance between a and b based on the chosen distance metric
        """
        if self.distance_type == "euclidean":
            print(a.size(), b.size())
            distance = torch.cdist(a, b, p = 2)
            return distance.squeeze(1)
        elif self.distance_type == "cosine":
            sim = F.cosine_similarity(a, b)
            return (1 - sim) / (1 + sim + 1e-5)
        else:
            return torch.cdist(a, b)
        
    def build_goal_list(self, path):
        images = []
        #TODO change hardcoding
        path_to_images = ASSETS_PATH + self.env_name + "/"
        flag = FLAG_DICT[self.image_type]
        for p in os.listdir(path_to_images):
            print(p)
            if flag in p:
                img = np.array(Image.open(path_to_images + p))
                images.append(img)
        # embed images
        image_embeddings = self.embedding_model.get_image_embedding(np.stack(images, axis=0))
        image_embeddings = utils.pool_images(images=image_embeddings, pooling_type=self.pooling)
        return image_embeddings
    
    def get_current_goal_embedding(self):
        # 3 if there is only one image
        if self.goals.ndim == 3:
            return self.goals
        else:
            return self.goals[0]
        