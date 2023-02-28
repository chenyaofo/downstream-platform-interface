

``` python
class Model:
    # interface of big model while tranferring to downstream tasks
    
    def get_description(self):
        # return the detailed info of this pretrained model, such as pretrained data, 
        # refered paper and a berif introdudction of this model
        pass

    @classmethod
    def from_pretrained(cls, name:str, load_weights:str) -> Model:
        pass
    
    def get_input_resolution(self):
        # return the input resolution of the pretraining stage
        # e.g., get_input_resolution() -> (3,224,224)
        pass
    
    def get_features_shape(self, downsampling_coefficient=None):
        # if the input size is the same as self.get_input_resolution(), 
        # return the shape of features based on the given downsampling_coefficient
        # if downsampling_coefficient is None, this func should return the shape of last layer features
        pass

    def has_features(self, downsampling_coefficient=None):
        # if this model can provide 1/downsampling_coefficient resolution features, return True; return False otherwise
        # if downsampling_coefficient is None, this func always return True
        # e.g., has_low_resolution_features(downsampling_coefficient=4) -> True
        pass

    def fetch_features(self, downsampling_coefficient):
        # after foward, user can get the 1/downsampling_coefficient resolution features via this func
        # if downsampling_coefficient is None, user can get the features from the last layer via this func
        # note that, every foward would flash the features buffer, you only can fecth the most recent features via this func
        # e.g., fetch_low_resolution_features(downsampling_coefficient=4) -> Tensor(size=[batch,64,56,56])
        pass

    def forward(self, x):
        pass
```
