# 该文档介绍如何改造大模型使得其符合标准

## 一、安装教程

请查看[安装教程](https://github.com/chenyaofo/downstream-platform-interface/blob/main/python/README.md)

## 二、编写exported_model.py和hubconf.py文件，继承大模型接口抽象类并实现其抽象接口函数

请查看大模型接口抽象类、抽象函数、及其注释[model_abc.py](https://github.com/chenyaofo/downstream-platform-interface/blob/main/python/model_downstream_interface/model_abc.py)，大致了解每个抽象函数的功能。

接下来，将结合改造[swin-transformer](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example/swin-transformer) 和 [vision-transformer](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example/vision-transformer) 这两个具体例子，介绍如何继承大模型接口抽象类并实现其抽象接口函数。

注意，用户不需要修改原始模型定义文件，只需要新增exported_model.py和hubconf.py文件。具体的文件目录结构如下([目录参考](https://github.com/chenyaofo/downstream-platform-interface/tree/main/example))：

--swin-transformer             // **把该文件夹打包为.zip压缩包**，即满足大模型接口规范，可把压缩包上传至平台进行大模型入仓校验。<br>
   --swin_transformer_tiny        // **该文件夹必须有唯一的命名**，不能和其他预训练大模型的文件夹命名重复，否则会引起导入失败。<br>
      --__init__.py                   // python包初始化文件，需在里面导入exported_model.py中定义的预训练大模型接口类。<br>
      --exported_model.py             // 需要用户编写的预训练大模型接口类定义文件，文件命名无强制要求。<br>
      --model.py                      // 预训练大模型原本的定义文件，文件命名无强制要求（如果模型定义较复杂，可用多个.py文件定义）。<br>
   --hubconf.py                   // **该文件必须命名为hubconf.py，否则系统无法识别。**<br>
   --weights                      // 存放模型参数和模型图的文件夹。<br>
   --swin_t-704ceda3.pth           // 模型参数。在exported_model.py的from_pretrained()函数中加载该文件。<br>


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
   





## 三、保存大模型





## 四、使用标准接口加载大模型，推理大模型并获取指定的特征