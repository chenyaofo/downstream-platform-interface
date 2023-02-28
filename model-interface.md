Based on the suggestions from plenty of downstream tasks, we design an interface of the big model in the platform as follows.
If someone seeks to put a big model into the platform, he/she should write the his/her big model code by inheriting the `Model` class.

``` python
import typing


class Model:
    # interface of big model while tranferring to downstream tasks

    def get_description(self):
        # return the detailed info of this pretrained model, such as pretrained data,
        # refered paper and a berif introdudction of this model
        pass

    @classmethod
    def from_pretrained(cls, load_weights: str) -> Model:
        # it should return a Model instance which loads pretrained weights according to the params
        pass

    def get_input_resolution(self) -> typing.Tuple[int]:
        # return the input resolution of the pretraining stage
        # e.g., get_input_resolution() -> (3,224,224)
        pass

    def get_features_shape(self, downsampling_coefficient: int = None):
        # if the input size is the same as self.get_input_resolution(),
        # return the shape of features based on the given downsampling_coefficient
        # if downsampling_coefficient is None, this func should return the shape of last layer features
        # e.g., get_features_shape(downsampling_coefficient=4) -> (64,56,56)
        pass
        
    def is_hierarchical_features(self):
        # if this model has hierarchical features (e.g., resnet and swin transformer), return True;
        # otherwise (e.g., vision transformer), return False
        pass
    
    def has_features(self, downsampling_coefficient: int = None):
        # if this model can provide 1/downsampling_coefficient resolution features, return True; return False otherwise
        # if downsampling_coefficient is None, this func always return True
        # e.g., has_low_resolution_features(downsampling_coefficient=4) -> True
        pass

    def fetch_features(self, downsampling_coefficient: int = None):
        # after foward, user can get the 1/downsampling_coefficient resolution features via this func
        # if downsampling_coefficient is None, user can get the features from the last layer via this func
        # note that, every foward would flash the features buffer, you only can fecth the most recent features via this func
        # impl hint: it can be impl via foward hooks and a contrainer (e.g., dict or list) of this instance
        # e.g., fetch_low_resolution_features(downsampling_coefficient=4) -> Tensor(size=[batch,64,56,56])
        pass

    def forward(self, x):
        pass

```

Another One:

```
import typing

class LayerIndexNotFoundException(Exception):
    pass

class Model:
    # interface of big model while tranferring to downstream tasks

    def get_description(self):
        # return the detailed info of this pretrained model, such as pretrained data,
        # refered paper and a berif introdudction of this model
        pass

    @classmethod
    def from_pretrained(cls, load_weights: str) -> Model:
        # it should return a Model instance which loads pretrained weights according to the params
        pass

    def get_depth(self) -> int:
        # return the number of layers in this model
        pass

    def _check_layer_index_range(self, layer_index: typing.Union[int, typing.Sequence[int]]):
        if isinstance(layer_index, int):
            layer_index = [layer_index]

        available_layer_indexes = list(range(self.get_depth()))
        for i in layer_index:
            if i not in available_layer_indexes:
                raise LayerIndexNotFoundException(f"Given layer index {i} is not available.")

    def get_features_shape(self, input_shape: typing.Tuple[int]=None, layer_index: int = None):
        # based on the input sample shape, return the feature shape at layer of given index
        # if the layer index is out of range, it would raise LayerIndexNotFoundException
        # e.g., get_features_shape(input_shape=(3,224,224), layer_index=6) -> (64,56,56)
        pass
    
    def get_features_shapes_all_layers(self, input_shape: typing.Tuple[int]=None):
        return [
            self.get_features_shape(input_shape, layer_id) for layer_id in range(self.get_depth())
        ]
        
    def is_hierarchical_features(self) -> bool:
        # if this model has hierarchical features (e.g., resnet and swin transformer), return True;
        # otherwise (e.g., vision transformer), return False
        pass
    
    def register_features_buffer_by_layer_indexes(self, layer_indexes: typing.Tuple[int]) -> None:
        # user can use this func tell the model which layers features would be stored in a buffer
        # only which a layer index is registered, user can fetch the features
        # if the layer index is out of range, it would raise LayerIndexNotFoundException
        pass

    def fetch_features(self) -> typing.Tuple[Tensor]:
        # after foward, user can get the features of given layers (specificed by register_features_buffer_by_layer_indexes) via this func
        # impl hint: it can be impl via foward hooks and a contrainer (e.g., dict or list) of this instance
        # e.g., fetch_low_resolution_features() -> Tuple[Tensor(size=[batch,768,1024])]
        pass

    def forward(self, x):
        pass


def get_pyramid_feature_layer_indexes(shapes_across_layers: typing.Tuple[typing.Tuple[int]]) -> typing.Tuple[int]:
    # this is a helper func, help user to select the indexes of last layer on every pyramid
    pass

# examples fo an object dtection task
big_model:Model = Model.from_pretrained("path/to/checkpoint")

big_model.register_features_buffer_by_layer_indexes(
    get_pyramid_feature_layer_indexes( # maybe here, user can select layers via other rules
        big_model.get_features_shapes_all_layers()
    )
)

for data, labels in dataloader:
    big_model(data)

    features: typing.Tuple[Tensor] = big_model.fetch_features()

    # you can use the features from big_model to perform distillation, tranferring and etc.
```
