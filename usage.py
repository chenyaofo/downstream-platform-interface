import os
import torch
import model_downstream_interface
from model_downstream_interface.utils import get_pyramid_feature_layer_indexes

# usage for vision transformer
print("vision transformer")
model: model_downstream_interface.BigModel4DownstreamInterface \
    = model_downstream_interface.load_model(
        os.path.abspath("example/vision-transformer")
    )

all_layer_indexed = list(range(model.get_depth()))

for i in all_layer_indexed:
    print(f"feature shape in layer {i}", model.get_features_shape(input_shape=(3, 224, 224), layer_index=i))

# here, we only select the last 4 layers
model.register_features_buffer_by_layer_indexes(all_layer_indexed[-4:])

x = torch.rand(1, 3, 224, 224)

model(x)

features = model.fetch_features()

for f in features:
    print("Actual fetched tesnor shape", f.shape)

# usage for swin transformer
print("-"*50 + "\n" + "swin transformer")
model: model_downstream_interface.BigModel4DownstreamInterface \
    = model_downstream_interface.load_model(
        os.path.abspath("example/swin-transformer")
    )


all_layer_indexed = list(range(model.get_depth()))

for i in all_layer_indexed:
    print(f"feature shape in layer {i}", model.get_features_shape(input_shape=(3, 224, 224), layer_index=i))

# here, we select layer based on a pyramid way
model.register_features_buffer_by_layer_indexes(
    get_pyramid_feature_layer_indexes(
        model.get_features_shapes_all_layers(input_shape=(3, 224, 224))
    )
)

x = torch.rand(1, 3, 224, 224)

model(x)

features = model.fetch_features()

for f in features:
    print("Actual fetched tesnor shape", f.shape)
