import abc
import typing

Tensor = typing.TypeVar("Tensor")


class LayerIndexNotFoundException(Exception):
    pass


class BigModel4DownstreamInterface(metaclass=abc.ABCMeta):
    # interface of big model while tranferring to downstream tasks

    @abc.abstractmethod
    def get_description(self):
        # return the detailed info of this pretrained model, such as pretrained data,
        # refered paper and a berif introdudction of this model
        pass

    @abc.abstractclassmethod
    def from_pretrained(cls, load_weights: str = None):
        # it should return a Model instance which loads pretrained weights according to the params
        pass

    @abc.abstractmethod
    def get_depth(self) -> int:
        # return the number of layers in this model
        pass

    def _check_layer_index_range(self, layer_index: typing.Union[int, typing.Sequence[int]]):
        if isinstance(layer_index, int):
            layer_index = [layer_index]

        available_layer_indexes = list(range(self.get_depth()))
        for i in layer_index:
            if i not in available_layer_indexes:
                raise LayerIndexNotFoundException(f"Given layer index {i} is out of range.")

    @abc.abstractmethod
    def get_features_shape(self, input_shape: typing.Sequence[int] = None, layer_index: int = None):
        # based on the input sample shape, return the feature shape at layer of given index
        # if the layer index is out of range, it would raise LayerIndexNotFoundException
        # e.g., get_features_shape(input_shape=(3,224,224), layer_index=6) -> (64,56,56)
        pass

    def get_features_shapes_all_layers(self, input_shape: typing.Sequence[int] = None):
        return [
            self.get_features_shape(input_shape, layer_id) for layer_id in range(self.get_depth())
        ]

    @abc.abstractmethod
    def is_hierarchical_features(self) -> bool:
        # if this model has hierarchical features (e.g., resnet and swin transformer), return True;
        # otherwise (e.g., vision transformer), return False
        pass
    
    @abc.abstractmethod
    def register_features_buffer_by_layer_indexes(self, layer_indexes: typing.Sequence[int]) -> None:
        # user can use this func tell the model which layers features would be stored in a buffer
        # only which a layer index is registered, user can fetch the features
        # if the layer index is out of range, it would raise LayerIndexNotFoundException
        pass

    @abc.abstractmethod
    def fetch_features(self) -> typing.Sequence[Tensor]:
        # after foward, user can get the features of given layers (specificed by register_features_buffer_by_layer_indexes) via this func
        # impl hint: it can be impl via foward hooks and a contrainer (e.g., dict or list) of this instance
        # e.g., fetch_low_resolution_features() -> Tuple[Tensor(size=[batch,768,1024])]
        pass
