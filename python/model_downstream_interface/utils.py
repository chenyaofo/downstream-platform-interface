import typing


def get_pyramid_feature_layer_indexes(shapes_across_layers: typing.Sequence[typing.Sequence[int]]) -> typing.Sequence[int]:
    # this is a helper func, help user to select the indexes of last layer on every pyramid
    pyramid_feature_layer_indexes = []
    last_layer_index = len(shapes_across_layers) - 1

    for i, _ in enumerate(shapes_across_layers):
        if i == last_layer_index:
            break
        if shapes_across_layers[i] != shapes_across_layers[i+1]:
            pyramid_feature_layer_indexes.append(i)
    
    
    if last_layer_index not in pyramid_feature_layer_indexes:
        pyramid_feature_layer_indexes.append(last_layer_index)
    return pyramid_feature_layer_indexes
