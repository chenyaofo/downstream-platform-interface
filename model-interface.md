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
