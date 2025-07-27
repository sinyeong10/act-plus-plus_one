from .backbone_models.backbone_registry import backbone_registry
from util.misc import NestedTensor, is_main_process

from torch import nn

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        #주어진 모듈의 순차적 실행

    def forward(self, tensor_list: NestedTensor):
        # print(tensor_list.shape) #torch.Size([8, 3, 480, 640])
        xs = self[0](tensor_list) 
        # print("\n\nxs",type(xs)) #xs <class 'collections.OrderedDict'>
        # for key, value in xs.items():
        #     print(f"{key}: {value.shape}") #0: torch.Size([8, 512, 15, 20])
        #0: torch.Size([8, 512, 15, 20]) : batchsize, num_channels, height, width
        out: List[NestedTensor] = [] #자료형의 명시적 표현
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            # print("\n\nposition encoding", type(self[1](x).to(x.dtype))) #<class 'torch.Tensor'>
            # print(self[1](x).to(x.dtype).shape) #torch.Size([1, 512, 15, 20])
            pos.append(self[1](x).to(x.dtype)) #torch.Size([1, 512, 15, 20])

        return out, pos

def build_backbone(args):
    print("\n\nargs.backbone", args.backbone)
    if args.backbone not in backbone_registry:
        raise ValueError(f"Invalid backbone '{args.backbone}'. Available: {list(backbone_registry)}")

    BackboneClass = backbone_registry[args.backbone]

    if getattr(BackboneClass, "_is_experimental", False):
        print(f"[WARNING] Using experimental backbone: {args.backbone}")

    backbone = BackboneClass(
            name=args.backbone,
            train_backbone=train_backbone,
            return_interm_layers=return_interm_layers,
            dilation=args.dilation,
            check_featuremap=args.feature_map
        )


    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model