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

                