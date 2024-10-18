import torch
from point_transformer.model import Backbone
from point_transformer.model import PointTransformerSeg
from point_transformer.model import PointTransformerCls
import omegaconf
import hydra

@hydra.main(config_path='config/PointTransformer.yaml')
class test():
    def __init__(self, args):
        print(f"This is a test file, args is {args}")
    

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


input = torch.randn(1, 1024, 3)
@hydra.main(config_path='config/PointTransformer.yaml')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    args.input_dim = 3 + 3
    args.num_class = 3
    num_category = 3
    num_part = args.num_class
    
    backbone = Backbone(args)
    pt_seg = PointTransformerSeg(args)
    pt_cls = PointTransformerCls(args)

    test = to_categorical(torch.tensor(1), num_category).repeat(1, input.shape[1], 1)
    input_feature = torch.cat([input, test], -1)

    # output = backbone(input_feature)

    output = pt_seg(input_feature)
    seg_pred = output.contiguous().view(-1, num_part)
    pred_choice = seg_pred.data.max(1)[1]
    
    # output = pt_cls(input_feature)
    print(output.shape)

def test2():
    pc = torch.randn(1, 3, 8)
    m = torch.max(torch.sqrt(torch.sum(pc**2, axis = 1)))
    print('test')


if __name__ == '__main__':
    # a =  test()
    # print("test")
    # main()
    import yaml

    with open("./config/PointTransformer.yaml", 'r') as stream:
        args = yaml.safe_load(stream)

    from collections import namedtuple

    def convert_to_namedtuple(dictionary):
        """Converts a dictionary to a namedtuple."""
        return namedtuple('GenericDict', dictionary.keys())(**dictionary)

    my_dict = args
    my_namedtuple = convert_to_namedtuple(my_dict)

    print(my_namedtuple)  # Output: Alice
