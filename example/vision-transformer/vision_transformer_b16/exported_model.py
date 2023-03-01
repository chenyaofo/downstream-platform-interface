import os
import typing
import functools

import torch

from .model import VisionTransformer, EncoderBlock
from model_downstream_interface import BigModel4DownstreamInterface


class VisionTransformer_B16(VisionTransformer, BigModel4DownstreamInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_entries = [
            module for _, module in self.named_modules() if isinstance(module, EncoderBlock)
        ]
        self.hooks = list()
        self.feature_buffers = dict()

    def get_description(self):
        return '''
                This model is trained on ImageNet-1k dataset with image resolution 224 following DeIT's training recipe (https://arxiv.org/abs/2012.12877).
                Aftering training, it achieves 81.072% top-1 accuracy and 95.318% top-5 accuracy.
                Please check https://arxiv.org/abs/2010.11929 for more details for the architecture of this model.
                '''

    @classmethod
    def from_pretrained(cls, load_weights: str = None, *args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        model = cls(
            image_size=image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            *args,
            **kwargs
        )

        load_weights = os.path.join(
            os.path.dirname(__file__),
            "..",
            "weights",
            "vit_b_16-c867db91.pth"
        ) if load_weights is None else load_weights  # here if load_weights is none, we use default ptraining weights

        model.load_state_dict(torch.load(load_weights, map_location="cpu"))

        return model

    def get_depth(self) -> int:
        return len(self.feature_entries)

    def get_features_shape(self, input_shape: typing.Sequence[int] = None, layer_index: int = None):
        if len(input_shape) != 3:
            raise ValueError(f"input_shape should have 3 dimensions, but got {input_shape}")
        if input_shape[0] != 3:
            raise ValueError(f"input should be a 3-channels images, but got {input_shape[0]} channels")
        if input_shape[1] != input_shape[2]:
            raise ValueError(f"the width and height of input image should be the same, but got {input_shape[1:]}")
        if input_shape[-1] % self.patch_size != 0:
            raise ValueError("Input shape indivisible by patch size")

        self._check_layer_index_range(layer_index)

        return (
            (input_shape[-1] // self.patch_size)**2 + 1,
            self.hidden_dim,
        )  # features in vision transformer has the same shape in every layer

    def is_hierarchical_features(self) -> bool:
        return False

    def register_features_buffer_by_layer_indexes(self, layer_indexes: typing.Sequence[int]) -> None:
        # if user register for the second time, we should clear existing hooks and feature buffers
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.feature_buffers.clear()

        self._check_layer_index_range(layer_indexes)
        self.layer_indexes = layer_indexes

        def save_features_hook(m, in_features, out_features, layer_index: int, feature_buffers: dict):
            feature_buffers[layer_index] = out_features

        for index, module in enumerate(self.feature_entries):
            if index in layer_indexes:
                self.hooks.append(
                    module.register_forward_hook(
                        functools.partial(save_features_hook, layer_index=index, feature_buffers=self.feature_buffers)
                    )
                )

    def fetch_features(self) -> typing.Sequence[torch.Tensor]:
        if not hasattr(self, "layer_indexes"):
            raise ValueError("Please call register_features_buffer_by_layer_indexes before fetch_features")
        return [
            self.feature_buffers[key] for key in self.layer_indexes
        ]

    def forward(self, x: torch.Tensor):
        # here, we slightly modify original forward function, make it run without classfier head

        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        return x
