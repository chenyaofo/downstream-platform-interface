import os
import typing
import functools

import torch

from .model import SwinTransformer, SwinTransformerBlock
from model_downstream_interface import BigModel4DownstreamInterface


class SwinTransformer_Tiny(SwinTransformer, BigModel4DownstreamInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_entries = [
            module for _, module in self.named_modules() if isinstance(module, SwinTransformerBlock)
        ]
        self.hooks = list()
        self.feature_buffers = dict()

    def get_description(self):
        return '''
                This model is trained on ImageNet-1k dataset with image resolution 224.
                Aftering training, it achieves 81.474% top-1 accuracy and 95.776% top-5 accuracy.
                Please check https://arxiv.org/abs/2103.14030 for more details for the architecture of this model.
                '''

    @classmethod
    def from_pretrained(cls, load_weights: str = None, *args, **kwargs):
        # image_size = kwargs.pop("image_size", 224)
        model = cls(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            stochastic_depth_prob=0.2,
            *args,
            **kwargs
        )

        load_weights = os.path.join(
            os.path.dirname(__file__),
            "..",
            "weights",
            "swin_t-704ceda3.pth"
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

        self._check_layer_index_range(layer_index)

        shapes = []
        for i_stage in range(len(self.depths)):
            dim = self.embed_dim * 2**i_stage
            resolution = input_shape[-1] // (self.patch_size[0]*2**i_stage)
            for i_layer in range(self.depths[i_stage]):
                shapes.append((resolution, resolution, dim))

        return shapes[layer_index]

    def is_hierarchical_features(self) -> bool:
        return True

    def register_features_buffer_by_layer_indexes(self, layer_indexes: typing.Sequence[int]) -> None:
        # if user register for the second time, we should clear existing hooks and feature buffers
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.feature_buffers.clear()

        self._check_layer_index_range(layer_indexes)
        self.layer_indexes = layer_indexes

        class SaveFeaturesHook:
            def __init__(self, feature_buffer:dict, layer_index:int):
                self.feature_buffer = feature_buffer
                self.layer_index = layer_index

            def excute(self, m, in_features, out_features):
                self.feature_buffers[self.layer_index] = out_features

        # def save_features_hook(m, in_features, out_features, layer_index: int, feature_buffers: dict):
        #     feature_buffers[layer_index] = out_features

        for index, module in enumerate(self.feature_entries):
            if index in layer_indexes:
                self.hooks.append(
                    module.register_forward_hook(
                        # functools.partial(save_features_hook, layer_index=index, feature_buffers=self.feature_buffers)
                        SaveFeaturesHook(
                            feature_buffer=self.feature_buffers,
                            layer_index=index
                        ).excute
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

        x = self.features(x)
        return x
