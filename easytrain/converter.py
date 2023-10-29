import torch
from torch import onnx


def convert_to_onnx(model: torch.nn.Module, save_path: str, model_args: tuple, model_kwargs: dict):
    """Using TorchDynamo to capture an FX graph

    ONNX
    ONNX Runtime
    ONNX Script

    For extending the operator, use the onnx registry
    """
    export_options = onnx.ExportOptions(
        dynamic_shapes=True
    )
    onnx.dynamo_export(
        model, 
        *model_args, 
        **model_kwargs, 
        export_options=export_options
    ).save(save_path)



if __name__ == '__main__':
    from torchvision.models import resnet18
    model = resnet18()
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    # (x, bias=None) -> {"bias": 2.}
    args = (x,)
    kwargs = {}     
    
    convert_to_onnx(model, "model.onnx", args, kwargs)