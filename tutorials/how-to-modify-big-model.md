# 该文档介绍如何改造大模型使得其符合标准

## 一、安装教程

请查看[安装教程](https://github.com/chenyaofo/downstream-platform-interface/blob/main/python/README.md)

## 二、编写exported_model.py和hubconf.py文件，继承大模型接口抽象类并实现其抽象接口函数

请查看大模型接口抽象类、抽象函数、及其注释[model_abc.py](https://github.com/chenyaofo/downstream-platform-interface/blob/main/python/model_downstream_interface/model_abc.py)，大致了解每个抽象函数的功能。

接下来，将结合改造Pytorch框架中的[swin-transformer](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example/swin-transformer) 和 [vision-transformer](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example/vision-transformer) 这两个具体例子，介绍如何继承大模型接口抽象类并实现其抽象接口函数。

注意，用户不需要修改原始模型定义文件，只需要新增exported_model.py和hubconf.py文件。具体的文件目录结构如下([目录参考](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example))：

```
-swin-transformer          // 把该文件夹打包为.zip压缩包，即满足大模型接口规范，可把压缩包上传至平台进行大模型入仓校验。
---swin_transformer_tiny   // 该文件夹必须有唯一的命名，不能和其他预训练大模型的文件夹命名重复，否则会引起导入失败。
-----__init__.py           // python包初始化文件，需在里面导入exported_model.py中定义的预训练大模型接口类。
-----exported_model.py     // 需要用户编写的预训练大模型接口类定义文件，文件命名无强制要求。
-----model.py              // 预训练大模型原本的定义文件，命名无要求（如模型定义复杂，可用多个.py文件定义）。
---hubconf.py              // 该文件必须命名为hubconf.py，否则系统无法识别。
---weights                 // 存放模型参数和模型图的文件夹。
-----swin_t-704ceda3.pth   // 模型参数。在exported_model.py的from_pretrained()函数中加载该文件。
```

1. 以[swin-transformer](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example/swin-transformer)为例，先介绍如何编写
   [hubconf.py](https://github.com/chenyaofo/downstream-platform-interface/blob/main/example/swin-transformer/hubconf.py)。
   
   填写dependencies列表，即大模型推理所用到的所有python包的列表。代码会自动检查能否导入列表中的包，如导入失败，则报错。
   ```
   dependencies = ["torch", "torchvision"]
   ```
   
   import [exported_model.py](https://github.com/chenyaofo/downstream-platform-interface/blob/main/example/swin-transformer/swin_transformer_tiny/exported_model.py)中，用户实现的SwinTransformer_Tiny类。该类继承了SwinTransformer, BigModel4DownstreamInterface两个父类。
   其中，SwinTransformer是待入仓的大模型定义类，而BigModel4DownstreamInterface则是[model_abc.py](https://github.com/chenyaofo/downstream-platform-interface/blob/main/python/model_downstream_interface/model_abc.py)中的大模型接口抽象类。
   ```
   from swin_transformer_tiny import SwinTransformer_Tiny
   ```
   
   重命名SwinTransformer_Tiny为BIG_PRETRAINED_MODEL，以方便后续统一调用。
   ```
   BIG_PRETRAINED_MODEL = SwinTransformer_Tiny
   ```

2. 以[swin-transformer](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example/swin-transformer)为例，介绍如何编写 
   [exported_model.py](https://github.com/chenyaofo/downstream-platform-interface/blob/main/example/swin-transformer/swin_transformer_tiny/exported_model.py)。
   **注意，样例中的模型所用深度学习框架为PyTorch，因此实现过程使用了PyTorch提供的一些函数。若要使用其他框架的预训练大模型，请自行调用具有对应功能的函数。**
   
   从model_downstream_interface导入BigModel4DownstreamInterface抽象类，从.model导入预训练模型的原始类。
   ```
   import os
   import typing
   import functools
   import torch   
   from .model import SwinTransformer, SwinTransformerBlock
   from model_downstream_interface import BigModel4DownstreamInterface
   ```

   定义SwinTransformer_Tiny类（类名用户自定，但注意需要在hubconf.py中通过BIG_PRETRAINED_MODEL = SwinTransformer_Tiny把类名赋值给BIG_PRETRAINED_MODEL），继承预训练模型的原始类SwinTransformer和大模型接口抽象类BigModel4DownstreamInterface。

   初始化self.feature_entries列表，保存所有SwinTransformerBlock对象。
   
   初始化self.hooks列表，用于存储获取特征所需的hook对象
   
   初始化self.feature_buffers字典，用于存储模型推理时的中间层特征
   
   ```
   class SwinTransformer_Tiny(SwinTransformer, BigModel4DownstreamInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.feature_entries = [
            module for _, module in self.named_modules() if isinstance(module, SwinTransformerBlock)
        ]
        self.hooks = list()
        self.feature_buffers = dict()
   ```

   返回对模型的描述
   ```
    def get_description(self):
        return '''
               This model is trained on ImageNet-1k dataset with image resolution 224.
               Aftering training, it achieves 81.474% top-1 accuracy and 95.776% top-5 accuracy.
               Please check https://arxiv.org/abs/2103.14030 for more details for the architecture of this model.
               '''
   ```

   该函数用于加载预训练模型。其中“cls”为class SwinTransformer_Tiny本身。

   首先通过 model = cls(xxx) 实例化一个SwinTransformer_Tiny对象。

   然后，用户需在该函数中显式给出预训练模型的参数的路径（如样例中的“swin_t-704ceda3.pth”），确保运行该函数即可加载预训练模型参数。

   最后，使用model.load_state_dict(torch.load(load_weights, map_location="cpu"))把模型参数导入至model对象中并返回

   ```
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
   ```

   返回self.feature_entries的长度
   ```
    def get_depth(self) -> int:
        return len(self.feature_entries)
   ```

   函数输入为“模型输入图像的尺寸”，以及“用户指定的layer_index（也就是feature_entries的index）”，返回该index所对应的特征的尺寸
   ```
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
   ```

   根据该预训练模型是否存在金字塔特征结构，直接返回True或者False。例如，Swim-transformer存在不同长宽的feature map，拥有金字塔结构的特征。而Vision-transformer的feature map的长和宽不随网络深度变化而变化，故不存在金字塔结构的特征。
   
   ```
    def is_hierarchical_features(self) -> bool:
        return True
   ```
   
   为对应的模块注册钩子hook，以在模型推理时通过钩子hook获取模型的中间特征。函数的输入是“需要注册钩子获取中间特征的layer/feature_entries/模块的index”
   
   定义save_features_hook函数，让钩子hook在模型推理时自动把该层输出特征保存到字典feature_buffers中。

   ```
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
   ```





## 三、保存大模型





## 四、使用标准接口加载大模型，推理大模型并获取指定的特征