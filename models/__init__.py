from .alexnet import *
from .densenet import *
from .mobilenet import *
from .resnet import *
from .shiftnet import *
from .vgg import *

from copy import deepcopy

def convert_model(model, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                  d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                  data_stationary):

    # copy the model, since we modify it internally
    model = deepcopy(model)

    if isinstance(model, ShiftNet):
        return convert_shiftnet19(model, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                                  d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                                  data_stationary)
    elif isinstance(model, AlexNet):
        return convert_alexnet(model, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                               d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                               data_stationary)
    elif isinstance(model, VGG):
        return convert_vgg(model, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                           d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                           data_stationary)
    elif isinstance(model, ResNet):
        return ConvertedResNet(model, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                               d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                               data_stationary)
    
    raise KeyError('Model: {} not found.', model.__class__.__name__)

def convert_value_model(model, w_move_terms, w_move_group, w_stat_values, w_stat_group,
                        d_move_terms, d_move_group, d_stat_values, d_stat_group,
                        data_stationary):
    # copy the model, since we modify it internally
    model = deepcopy(model)

    if isinstance(model, ShiftNet):
        return convert_value_shiftnet19(model, w_move_terms, w_move_group, w_stat_values,
                                        w_stat_group, d_move_terms, d_move_group,
                                        d_stat_values, d_stat_group, data_stationary)

    raise KeyError('Model: {} not found.', model.__class__.__name__)

def data_stationary_point(model):
    if isinstance(model, ShiftNet):
        return 12
    elif isinstance(model, AlexNet):
        return 6
    elif isinstance(model, VGG):
        return 12
    elif isinstance(model, ResNet):
        return 16

    raise KeyError('Model: {} not found.', model.__class__.__name__)