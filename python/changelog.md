# 版本变更日志

 - v0.1.0 (2023.03.01)
   - 添加了抽象类`BigModel4DownstreamInterface`作为下游训练大模型的接口
   - 添加了大模型加载入口`load_model`，用户可以调用此接口快速实例化
   - 添加了帮助函数`get_pyramid_feature_layer_indexes`，用户可根据此函数快速构建特征金字塔