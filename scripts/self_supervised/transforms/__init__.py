from . import neural as neural_transforms
from . import cifar
from . import graphs
from . import graph_random_crop
from . import xray
from . import EnhanceEdge
from . import RandErasingMean
from . import CustomAugs
from . import synapse


class GenerateViews:
    def __init__(self, *transforms):
        self.transforms = transforms

    @staticmethod
    def prepare_views(inputs):
        data_1, data_2 = inputs
        outputs = {'view_1': data_1, 'view_2': data_2}
        return outputs

    def __call__(self,data):
        return tuple([t(data) for t in self.transforms])
