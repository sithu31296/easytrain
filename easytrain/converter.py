import torch
import torch_tensorrt
from torch import onnx


def optimize_model(model: torch.nn.Module):
    return torch.compile(model, mode='reduce-overhead')


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


def convert_to_tensorrt_jit(model: torch.nn.Module, inputs: torch.Tensor):
    """Flexible + JIT

    https://www.youtube.com/watch?v=eGDMJ3MY4zk

    Only Python deployment

    torch.compile will split the model into Graph, Dynamo Guard, Graph

    Guard runs in python, Graph recompilation on conditional change or input shape change
    Graph is Torch-TRT Optimized Graph

    Compiled models cannot be serialized
    """
    optimized_model = torch.compile(
        model,
        fullgraph=False,
        dynamic=None,
        backend="tensorrt",
        options={
            "debug": True,
            "enabled_precisions": {torch.half},     # layer precision for tensorRT, FP32 and FP16 supported
            "min_block_size": 1,                    # minimum number of operators per TRT Engine block
            "torch_executed_ops": None,             # Operators required to run in PyTorch
            "optimization_level": None,             # TRT optimization level 0-5, higher level implies longer build time
            "truncate_long_and_dobule": True,       # Toggle truncation of 64-bit TRT engine inputs and weights to 32-bits
            "version_compatbile": None,             # Toggle version forward-compatibility for TRT engines
            "use_python_runtime": False,            # Toggle Python/C++ TensorRT runtime
        }
    )
    optimized_model(**inputs)
    return optimized_model


def convert_to_tensorrt_aot(model: torch.nn.Module):
    """Serializable + AOI
    Serialization supported
    Python or C++ deployment

    nn.Module -> export.ExportedProgram -> fx.GraphModule -> Serialize

    Supports in both compilation models
        * full model (all ops get converted to TensorRT)
        * hybrid model (mix of PyTorch and TensorRT ops)

    Serialized as TorchScript or ExportedProgram
    """
    # dynamic shape, tensorrt expects a range of shapes (min, opt, max) for each dynamic input in the graph
    inputs = [torch_tensorrt.Input(min_shape=(1, 512, 1, 1),    
                                   opt_shape=(4, 512, 1, 1),
                                   max_shape=(8, 512, 1, 1)
                                   )]

    # traces the graph with support for both static and dynamic inputs
    exp_program = torch_tensorrt.dynamo.trace(model, inputs)

    # performs lowering, partitioning and conversion to TensoRT
    # returns a graphmodule with TensorRT engines as attributes
    trt_model = torch_tensorrt.dynamo.compile(exp_program, inputs=inputs)

    trt_ser_model = torch_tensorrt.dynamo.serialize(trt_model, *inputs)

    torch.save(trt_ser_model, "trt_model.pt")
    return trt_model


if __name__ == '__main__':
    from torchvision.models import resnet18
    model = resnet18()
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    # (x, bias=None) -> {"bias": 2.}
    args = (x,)
    kwargs = {}     

    print(torch._dynamo.list_backends())
    
    # convert_to_onnx(model, "model.onnx", args, kwargs)