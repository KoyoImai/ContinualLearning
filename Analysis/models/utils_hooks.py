


class FeatureHook:
    def __init__(self):
        self.outputs = {}

    def hook_fn(self, name):
        def fn(module, input, output):
            self.outputs[name] = output.detach()
        return fn

    def clear(self):
        self.outputs = {}




def register_resnet18_hooks(model, opt, granularity="block"):
    """
    model.encoder にhookを登録して FeatureHook を返す。
    granularity: "block" | "basicblock" | "conv"
    """
    hooker = FeatureHook()

    if granularity == "block":
        hook_targets = {
            "conv1": model.encoder.conv1,
            "layer1": model.encoder.layer1,
            "layer2": model.encoder.layer2,
            "layer3": model.encoder.layer3,
            "layer4": model.encoder.layer4,
        }

    elif granularity == "basicblock":
        hook_targets = {}
        for i in range(1, 5):  # layer1〜4
            layer = getattr(model.encoder, f"layer{i}")
            for j, block in enumerate(layer):
                name = f"block_{(i - 1) * 2 + j + 1}"
                hook_targets[name] = block

    elif granularity == "conv":
        hook_targets = {}
        for i in range(1, 5):  # layer1〜4
            
            if opt.use_dp:
                layer = getattr(model.module.encoder, f"layer{i}")
            else:
                layer = getattr(model.encoder, f"layer{i}")
            
            for j, block in enumerate(layer):
                idx = (i - 1) * 2 + j + 1  # block index
                # conv1
                name1 = f"block_{idx}_conv1"
                hook_targets[name1] = block.conv1
                # conv2
                name2 = f"block_{idx}_conv2"
                hook_targets[name2] = block.conv2

    else:
        raise ValueError("granularity must be 'block', 'basicblock', or 'conv'")

    # register all
    for name, module in hook_targets.items():
        module.register_forward_hook(hooker.hook_fn(name))

    return hooker





# def register_resnet18_hooks(model, granularity="block"):
#     """
#     model.encoder は torchvision.models.resnet18 相当
#     granularity: "block" or "basicblock"
#     """
#     hooker = FeatureHook()

#     if granularity == "block":
#         hooker_dict = {
#             "conv1": model.encoder.conv1,
#             "layer1": model.encoder.layer1,
#             "layer2": model.encoder.layer2,
#             "layer3": model.encoder.layer3,
#             "layer4": model.encoder.layer4,
#         }
#         for name, module in hooker_dict.items():
#             module.register_forward_hook(hooker.hook_fn(name))

#     elif granularity == "basicblock":
#         for i in range(1, 5):  # layer1 to layer4
#             layer = getattr(model.encoder, f"layer{i}")
#             for j, block in enumerate(layer):
#                 name = f"block_{(i-1)*2 + j + 1}"  # block_1 to block_8
#                 block.register_forward_hook(hooker.hook_fn(name))
#     else:
#         raise ValueError("granularity must be 'block' or 'basicblock'")

#     return hooker


