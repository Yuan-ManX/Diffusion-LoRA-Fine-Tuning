import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


class LoraInjectedLinear(nn.Module):
    """
    LoraInjectedLinear 类是对 nn.Linear 的扩展，注入了 LoRA（低秩适应）机制。
    LoRA 通过在原始线性层的基础上添加低秩矩阵来减少可训练参数数量，从而加速训练和推理。
    """
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        # 检查 LoRA 的秩是否超过输入或输出特征的最小维度
        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        
        # 记录 LoRA 的秩
        self.r = r

        # 定义原始的线性层
        self.linear = nn.Linear(in_features, out_features, bias)

        # 定义 LoRA 的下层线性层，将输入特征映射到低秩空间
        self.lora_down = nn.Linear(in_features, r, bias=False)
        # 定义 Dropout 层，防止过拟合并增加模型的泛化能力
        self.dropout = nn.Dropout(dropout_p)
        # 定义 LoRA 的上层线性层，将低秩空间的表示映射回输出特征空间
        self.lora_up = nn.Linear(r, out_features, bias=False)
        # 定义缩放因子，用于调整 LoRA 的贡献
        self.scale = scale
        # 定义选择器（selector），这里使用恒等映射（Identity），后续可以自定义
        self.selector = nn.Identity()

        # 初始化 LoRA 下层线性层的权重，使用正态分布，标准差为 1/r
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        # 初始化 LoRA 上层线性层的权重为零
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        """
        前向传播函数，计算线性变换和 LoRA 调整的总和。
        
        参数:
            input (torch.Tensor): 输入张量，形状为 (batch_size, in_features)
        
        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)
        """
        # 计算原始线性层的输出
        # 计算 LoRA 的调整部分：
        # 1. 通过选择器（selector）处理输入
        # 2. 通过 LoRA 下层线性层映射到低秩空间
        # 3. 应用 Dropout
        # 4. 通过 LoRA 上层线性层映射回输出特征空间
        # 5. 乘以缩放因子
        # 返回原始线性层输出和 LoRA 调整部分的总和
        return (
            self.linear(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        """
        获取 LoRA 层的权重，以便进行权重合并或其他操作。
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 包含 LoRA 上层和下层线性层的权重
        """
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        """
        设置选择器为对角矩阵，仅在 LoRA 秩为1时有效。
        
        参数:
            diag (torch.Tensor): 1D 张量，形状为 (r,)
        """
        # 检查 diag 的形状是否与 LoRA 的秩匹配
        assert diag.shape == (self.r,)
        # 将选择器定义为线性层，初始化为对角矩阵
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = torch.diag(diag)
        # 将选择器的权重移动到与 LoRA 上层线性层相同的设备和类型
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


class LoraInjectedConv2d(nn.Module):
    """
    LoraInjectedConv2d 类是对 PyTorch 中 nn.Conv2d 层的扩展，注入了 LoRA（低秩适应）机制。
    LoRA 通过在原始卷积层的基础上添加低秩卷积层来减少可训练参数数量，从而加速训练和推理。
    
    参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int 或 tuple): 卷积核大小，可以是单个整数或两个整数的元组。
        stride (int 或 tuple, 可选): 卷积步幅，默认为1。
        padding (int 或 tuple, 可选): 输入的每条边补充0的层数，默认为0。
        dilation (int 或 tuple, 可选): 卷积核元素之间的间距，默认为1。
        groups (int, 可选): 输入通道和输出通道的分组数，默认为1。
        bias (bool, 可选): 是否使用偏置，默认为True。
        r (int, 可选): LoRA 的秩（rank），即低秩矩阵的维度，默认为4。
        dropout_p (float, 可选): Dropout 的概率，默认为0.1。
        scale (float, 可选): 缩放因子，用于调整 LoRA 的贡献，默认为1.0。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        # 检查 LoRA 的秩是否超过输入或输出通道的最小值
        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        
        # 记录 LoRA 的秩
        self.r = r

        # 定义原始的卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,           # 输入通道数
            out_channels=out_channels,         # 输出通道数
            kernel_size=kernel_size,           # 卷积核大小
            stride=stride,                     # 卷积步幅
            padding=padding,                   # 输入的每条边补充0的层数
            dilation=dilation,                 # 卷积核元素之间的间距
            groups=groups,                     # 输入通道和输出通道的分组数
            bias=bias,                         # 是否使用偏置
        )

        # 定义 LoRA 的下层卷积层，将输入通道映射到低秩空间
        self.lora_down = nn.Conv2d(
            in_channels=in_channels,           # 输入通道数
            out_channels=r,                    # 输出通道数为 LoRA 的秩
            kernel_size=kernel_size,           # 卷积核大小，与原始卷积层相同
            stride=stride,                     # 卷积步幅，与原始卷积层相同
            padding=padding,                   # 输入的每条边补充0的层数，与原始卷积层相同
            dilation=dilation,                 # 卷积核元素之间的间距，与原始卷积层相同
            groups=groups,                     # 分组数，与原始卷积层相同
            bias=False,                        # 不使用偏置
        )

        # 定义 Dropout 层，防止过拟合并增加模型的泛化能力
        self.dropout = nn.Dropout(dropout_p)

        # 定义 LoRA 的上层卷积层，将低秩空间的表示映射回输出通道空间
        self.lora_up = nn.Conv2d(
            in_channels=r,                     # 输入通道数为 LoRA 的秩
            out_channels=out_channels,         # 输出通道数与原始卷积层相同
            kernel_size=1,                     # 卷积核大小为1x1
            stride=1,                          # 卷积步幅为1
            padding=0,                         # 不进行填充
            bias=False,                        # 不使用偏置
        )

        # 定义选择器（selector），这里使用恒等映射（Identity），后续可以自定义
        self.selector = nn.Identity()
        # 定义缩放因子，用于调整 LoRA 的贡献
        self.scale = scale

        # 初始化 LoRA 下层卷积层的权重，使用正态分布，标准差为 1/r
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        # 初始化 LoRA 上层卷积层的权重为零
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        """
        前向传播函数，计算卷积变换和 LoRA 调整的总和。
        
        参数:
            input (torch.Tensor): 输入张量，形状为 (batch_size, in_channels, H, W)
        
        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_channels, H', W')
        """
        # 计算原始卷积层的输出
        # 计算 LoRA 的调整部分：
        # 1. 通过选择器（selector）处理输入
        # 2. 通过 LoRA 下层卷积层映射到低秩空间
        # 3. 应用 Dropout
        # 4. 通过 LoRA 上层卷积层映射回输出通道空间
        # 5. 乘以缩放因子
        # 返回原始卷积层输出和 LoRA 调整部分的总和
        return (
            self.conv(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        """
        获取 LoRA 层的权重，以便进行权重合并或其他操作。
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 包含 LoRA 上层和下层卷积层的权重
        """
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        """
        设置选择器为对角矩阵，仅在 LoRA 秩为1时有效。
        
        参数:
            diag (torch.Tensor): 1D 张量，形状为 (r,)
        
        异常:
            AssertionError: 如果 diag 的形状不等于 (r,)
        """
        # 检查 diag 的形状是否与 LoRA 的秩匹配
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        # 将选择器定义为1x1卷积层，初始化为对角矩阵
        self.selector = nn.Conv2d(
            in_channels=self.r,                # 输入通道数为 LoRA 的秩
            out_channels=self.r,               # 输出通道数也为 LoRA 的秩
            kernel_size=1,                     # 卷积核大小为1x1
            stride=1,                          # 卷积步幅为1
            padding=0,                         # 不进行填充
            bias=False,                        # 不使用偏置
        )
        # 将选择器的权重设置为对角矩阵
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        # 将选择器的权重移动到与 LoRA 上层卷积层相同的设备和类型
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

UNET_EXTENDED_TARGET_REPLACE = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

TEXT_ENCODER_EXTENDED_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    查找模型中所有属于特定类（或类的组合）的子模块。

    返回所有匹配的子模块，以及这些子模块的父模块和它们被引用的名称。

    参数:
        model (nn.Module): 需要搜索的 PyTorch 模型。
        search_class (List[Type[nn.Module]], 可选): 要搜索的模块类列表，默认为 [nn.Linear]。
    
    返回:
        Generator[Tuple[nn.Module, str, nn.Module]]: 一个生成器，生成包含父模块、名称和子模块的元组。
    """
    # 遍历模型中的每个父模块
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


def _find_modules_v2(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedLinear,
        LoraInjectedConv2d,
    ],
):
    """
    查找所有属于特定类（或类的组合）的模块，这些模块是其他特定类（或类的组合）模块的直接或间接后代。

    返回所有匹配的模块，以及这些模块的父模块和它们被引用的名称。

    参数:
        model (nn.Module): 需要搜索的 PyTorch 模型。
        ancestor_class (Optional[Set[str]], 可选): 祖先模块的类名集合，如果为 None，则遍历所有模块。
        search_class (List[Type[nn.Module]], 可选): 要搜索的模块类列表，默认为 [nn.Linear]。
        exclude_children_of (Optional[List[Type[nn.Module]]], 可选): 要排除的子模块类列表，默认为 [LoraInjectedLinear, LoraInjectedConv2d]。
    
    返回:
        Generator[Tuple[nn.Module, str, nn.Module]]: 一个生成器，生成包含父模块、名称和子模块的元组。
    """
    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def _find_modules_old(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [LoraInjectedLinear],
):
    ret = []
    for _module in model.modules():
        if _module.__class__.__name__ in ancestor_class:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__ in search_class:
                    ret.append((_module, name, _child_module))
    print(ret)
    return ret


_find_modules = _find_modules_v2


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    """
    将 LoRA 注入到模型中，并返回需要训练的 LoRA 参数组。

    参数:
        model (nn.Module): 需要注入 LoRA 的 PyTorch 模型。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 DEFAULT_TARGET_REPLACE。
        r (int, 可选): LoRA 的秩（rank），即低秩矩阵的维度，默认为4。
        loras (str, 可选): LoRA 权重文件的路径（.pt 文件），如果为 None，则不加载预训练的 LoRA 权重。
        verbose (bool, 可选): 是否打印详细信息，默认为 False。
        dropout_p (float, 可选): Dropout 的概率，默认为0.0。
        scale (float, 可选): 缩放因子，用于调整 LoRA 的贡献，默认为1.0。

    返回:
        Tuple[List[Parameter], List[str]]: 包含需要训练的 LoRA 参数组的列表和模块名称的列表。
    """
    # 初始化需要训练的参数列表和模块名称列表
    require_grad_params = []
    names = []

    # 如果提供了 LoRA 权重文件路径，则加载 LoRA 权重
    if loras != None:
        loras = torch.load(loras)

    # 遍历模型中所有需要替换的目标模块
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        # 获取当前线性层的权重和偏置
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection : injecting lora into ", name)
            print("LoRA Injection : weight shape", weight.shape)

        # 创建 LoraInjectedLinear 实例，替换原始的线性层
        _tmp = LoraInjectedLinear(
            _child_module.in_features,  # 输入特征的维度
            _child_module.out_features,  # 输出特征的维度
            _child_module.bias is not None,  # 是否使用偏置
            r=r,  # LoRA 的秩
            dropout_p=dropout_p,  # Dropout 的概率
            scale=scale,  # 缩放因子
        )

        # 将原始线性层的权重赋值给新的 LoRA 线性层
        _tmp.linear.weight = weight

        # 如果存在偏置，则将偏置也赋值给新的 LoRA 线性层
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        # 将新的 LoRA 线性层移动到与原始模块相同的设备和类型
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        # 替换模型中的原始模块为新的 LoRA 模块
        _module._modules[name] = _tmp

        # 将 LoRA 层的可训练参数添加到参数列表中
        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        # 如果提供了 LoRA 权重文件，则加载预训练的 LoRA 权重
        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)

        # 设置 LoRA 层的权重为可训练
        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        # 将模块名称添加到名称列表中
        names.append(name)

    # 返回需要训练的参数组和模块名称
    return require_grad_params, names


def inject_trainable_lora_extended(
    model: nn.Module,
    target_replace_module: Set[str] = UNET_EXTENDED_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
):
    """
    将 LoRA 注入到模型中，并返回需要训练的 LoRA 参数组。此函数支持线性层和卷积层。

    参数:
        model (nn.Module): 需要注入 LoRA 的 PyTorch 模型。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 UNET_EXTENDED_TARGET_REPLACE。
        r (int, 可选): LoRA 的秩（rank），即低秩矩阵的维度，默认为4。
        loras (str, 可选): LoRA 权重文件的路径（.pt 文件），如果为 None，则不加载预训练的 LoRA 权重。

    返回:
        Tuple[List[Parameter], List[str]]: 包含需要训练的 LoRA 参数组的列表和模块名称的列表。
    """
    # 初始化需要训练的参数列表和模块名称列表
    require_grad_params = []
    names = []

    # 如果提供了 LoRA 权重文件路径，则加载 LoRA 权重
    if loras != None:
        loras = torch.load(loras)

    # 遍历模型中所有需要替换的目标模块
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, nn.Conv2d]
    ):
        # 如果当前模块是线性层
        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias

            # 创建 LoraInjectedLinear 实例，替换原始的线性层
            _tmp = LoraInjectedLinear(
                _child_module.in_features,  # 输入特征的维度
                _child_module.out_features,  # 输出特征的维度
                _child_module.bias is not None,  # 是否使用偏置
                r=r,  # LoRA 的秩
            )

            # 将原始线性层的权重赋值给新的 LoRA 线性层
            _tmp.linear.weight = weight

            # 如果存在偏置，则将偏置也赋值给新的 LoRA 线性层
            if bias is not None:
                _tmp.linear.bias = bias
        
        # 如果当前模块是卷积层
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias

            # 创建 LoraInjectedConv2d 实例，替换原始的卷积层
            _tmp = LoraInjectedConv2d(
                _child_module.in_channels, # 输入通道数
                _child_module.out_channels, # 输出通道数
                _child_module.kernel_size, # 卷积核大小
                _child_module.stride, # 卷积步幅
                _child_module.padding, # 填充大小
                _child_module.dilation, # 扩张率
                _child_module.groups, # 分组数
                _child_module.bias is not None, # 是否使用偏置
                r=r, # LoRA 的秩
            )
            
            # 将原始卷积层的权重赋值给新的 LoRA 卷积层
            _tmp.conv.weight = weight
            # 如果存在偏置，则将偏置也赋值给新的 LoRA 卷积层
            if bias is not None:
                _tmp.conv.bias = bias

        # switch the module
        # 将新的 LoRA 模块移动到与原始模块相同的设备和类型
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        if bias is not None:
            _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

        # 替换模型中的原始模块为新的 LoRA 模块
        _module._modules[name] = _tmp

        # 将 LoRA 层的可训练参数添加到参数列表中
        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        # 如果提供了 LoRA 权重文件，则加载预训练的 LoRA 权重
        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)

        # 设置 LoRA 层的权重为可训练
        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        # 将模块名称添加到名称列表中
        names.append(name)

    # 返回需要训练的参数组和模块名称
    return require_grad_params, names


def extract_lora_ups_down(model, target_replace_module=DEFAULT_TARGET_REPLACE):
    """
    从模型中提取所有注入的 LoRA 层的上 (lora_up) 和下 (lora_down) 层。

    参数:
        model (nn.Module): 包含已注入 LoRA 层的 PyTorch 模型。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 DEFAULT_TARGET_REPLACE。
    
    返回:
        List[Tuple[nn.Module, nn.Module]]: 包含所有 LoRA 层的上 (lora_up) 和下 (lora_down) 层的元组列表。
    
    异常:
        ValueError: 如果模型中没有注入任何 LoRA 层，则抛出此异常。
    """
    # 初始化一个空列表，用于存储 LoRA 层的上 (lora_up) 和下 (lora_down) 层
    loras = []

    # 遍历模型中所有需要替换的目标模块，查找已注入的 LoRA 层
    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        # 将每个 LoRA 层的上 (lora_up) 和下 (lora_down) 层添加到列表中
        loras.append((_child_module.lora_up, _child_module.lora_down))

    # 如果没有找到任何 LoRA 层，则抛出 ValueError 异常
    if len(loras) == 0:
        raise ValueError("No lora injected.")

    # 返回包含所有 LoRA 层的上 (lora_up) 和下 (lora_down) 层的元组列表
    return loras


def extract_lora_as_tensor(
    model, target_replace_module=DEFAULT_TARGET_REPLACE, as_fp16=True
):
    """
    从模型中提取所有注入的 LoRA 层的上 (lora_up) 和下 (lora_down) 层，并将它们转换为张量。

    参数:
        model (nn.Module): 包含已注入 LoRA 层的 PyTorch 模型。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 DEFAULT_TARGET_REPLACE。
        as_fp16 (bool, 可选): 是否将权重转换为 float16 类型，默认为 True。
    
    返回:
        List[Tuple[torch.Tensor, torch.Tensor]]: 包含所有 LoRA 层的上 (lora_up) 和下 (lora_down) 层的张量元组列表。
    
    异常:
        ValueError: 如果模型中没有注入任何 LoRA 层，则抛出此异常。
    """
    # 初始化一个空列表，用于存储 LoRA 层的上 (lora_up) 和下 (lora_down) 层张量
    loras = []

    # 遍历模型中所有需要替换的目标模块，查找已注入的 LoRA 层
    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d], # 搜索的模块类为 LoraInjectedLinear 和 LoraInjectedConv2d
    ):
        # 获取 LoRA 层的上 (lora_up) 和下 (lora_down) 层的权重张量
        up, down = _child_module.realize_as_lora()
        # 如果需要，将权重转换为 float16 类型
        if as_fp16:
            up = up.to(torch.float16)
            down = down.to(torch.float16)

        # 将上 (lora_up) 和下 (lora_down) 层的权重张量添加到列表中
        loras.append((up, down))

    # 如果没有找到任何 LoRA 层，则抛出 ValueError 异常
    if len(loras) == 0:
        raise ValueError("No lora injected.")

    # 返回包含所有 LoRA 层的上 (lora_up) 和下 (lora_down) 层的张量元组列表
    return loras


def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module=DEFAULT_TARGET_REPLACE,
):
    """
    将模型中所有注入的 LoRA 层的权重保存到指定的文件中。

    参数:
        model (nn.Module): 包含已注入 LoRA 层的 PyTorch 模型。
        path (str, 可选): 保存 LoRA 权重的文件路径，默认为 "./lora.pt"。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 DEFAULT_TARGET_REPLACE。
    
    异常:
        ValueError: 如果模型中没有注入任何 LoRA 层，则抛出此异常。
    """
    # 初始化一个空列表，用于存储 LoRA 层的权重
    weights = []

    # 提取所有注入的 LoRA 层的上 (lora_up) 和下 (lora_down) 层
    for _up, _down in extract_lora_ups_down(
        model, target_replace_module=target_replace_module
    ):
        # 将上 (lora_up) 和下 (lora_down) 层的权重转换为 float16 类型并移动到 CPU
        weights.append(_up.weight.to("cpu").to(torch.float16))
        weights.append(_down.weight.to("cpu").to(torch.float16))

    # 将权重列表保存到指定的文件中
    torch.save(weights, path)


def save_lora_as_json(model, path="./lora.json"):
    """
    将模型中所有注入的 LoRA 层的权重保存为 JSON 格式的文件。

    参数:
        model (nn.Module): 包含已注入 LoRA 层的 PyTorch 模型。
        path (str, 可选): 保存 LoRA 权重的 JSON 文件路径，默认为 "./lora.json"。
    
    异常:
        ValueError: 如果模型中没有注入任何 LoRA 层，则抛出此异常。
    """
    # 初始化一个空列表，用于存储 LoRA 层的权重
    weights = []

    # 提取所有注入的 LoRA 层的上 (lora_up) 和下 (lora_down) 层
    for _up, _down in extract_lora_ups_down(model):
        # 将上 (lora_up) 和下 (lora_down) 层的权重转换为 NumPy 数组并转换为列表
        weights.append(_up.weight.detach().cpu().numpy().tolist())
        weights.append(_down.weight.detach().cpu().numpy().tolist())

    import json

    # 将权重列表写入指定的 JSON 文件中
    with open(path, "w") as f:
        json.dump(weights, f)


def save_safeloras_with_embeds(
    modelmap: Dict[str, Tuple[nn.Module, Set[str]]] = {},
    embeds: Dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    """
    将多个模块中的 LoRA 权重保存到一个单独的 safetensor 文件中。

    modelmap 是一个字典，格式为 {
        "模块名称": (模型, 目标替换模块集合)
    }

    参数:
        modelmap (Dict[str, Tuple[nn.Module, Set[str]]], 可选): 模型映射字典，默认为空字典。
        embeds (Dict[str, torch.Tensor], 可选): 嵌入字典，默认为空字典。
        outpath (str, 可选): 输出文件路径，默认为 "./lora.safetensors"。
    """
    # 初始化权重字典和元数据字典
    weights = {}
    metadata = {}

    # 遍历模型映射字典中的每个模块
    for name, (model, target_replace_module) in modelmap.items():
        # 将目标替换模块集合转换为 JSON 字符串并存储在元数据字典中
        metadata[name] = json.dumps(list(target_replace_module))

        # 提取所有注入的 LoRA 层的上 (lora_up) 和下 (lora_down) 层
        for i, (_up, _down) in enumerate(
            extract_lora_as_tensor(model, target_replace_module)
        ):
            # 获取 LoRA 层的秩（rank）
            rank = _down.shape[0]

            # 将秩存储在元数据字典中
            metadata[f"{name}:{i}:rank"] = str(rank)
            # 将上 (lora_up) 和下 (lora_down) 层的张量添加到权重字典中
            weights[f"{name}:{i}:up"] = _up
            weights[f"{name}:{i}:down"] = _down

    # 遍历嵌入字典中的每个 token
    for token, tensor in embeds.items():
        # 将嵌入标志存储在元数据字典中
        metadata[token] = EMBED_FLAG
        # 将嵌入张量添加到权重字典中
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    # 调用 safe_save 函数，将权重和元数据保存到指定的 safetensors 文件中
    safe_save(weights, outpath, metadata)


def save_safeloras(
    modelmap: Dict[str, Tuple[nn.Module, Set[str]]] = {},
    outpath="./lora.safetensors",
):
    """
    将多个模块中的 LoRA 权重保存到一个单独的 safetensor 文件中。

    参数:
        modelmap (Dict[str, Tuple[nn.Module, Set[str]]], 可选): 模型映射字典，默认为空字典。
        outpath (str, 可选): 输出文件路径，默认为 "./lora.safetensors"。
    """
    # 调用 save_safeloras_with_embeds 函数，嵌入字典为空字典
    return save_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def convert_loras_to_safeloras_with_embeds(
    modelmap: Dict[str, Tuple[str, Set[str], int]] = {},
    embeds: Dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    """
    将多个 PyTorch.pt 文件中的 LoRA 转换为单个 safetensor 文件。

    modelmap 是一个字典，格式为 {
        "模块名称": (PyTorch模型路径, 目标替换模块集合, LoRA秩)
    }

    参数:
        modelmap (Dict[str, Tuple[str, Set[str], int]], 可选): 模型映射字典，键为模块名称，值为 (PyTorch模型路径, 目标替换模块集合, LoRA秩) 的元组，默认为空字典。
        embeds (Dict[str, torch.Tensor], 可选): 嵌入字典，键为 token，值为对应的张量，默认为空字典。
        outpath (str, 可选): 输出文件路径，默认为 "./lora.safetensors"。
    """
    # 初始化权重字典和元数据字典
    weights = {}
    metadata = {}

    # 遍历模型映射字典中的每个模块
    for name, (path, target_replace_module, r) in modelmap.items():
        # 将目标替换模块集合转换为 JSON 字符串并存储在元数据字典中
        metadata[name] = json.dumps(list(target_replace_module))

        # 加载 LoRA 权重文件
        lora = torch.load(path)
        # 遍历 LoRA 权重列表
        for i, weight in enumerate(lora):
            # 判断当前权重是上 (lora_up) 还是下 (lora_down) 层
            is_up = i % 2 == 0
            # 计算索引
            i = i // 2

            if is_up:
                # 如果是上 (lora_up) 层，记录秩并存储权重
                metadata[f"{name}:{i}:rank"] = str(r)
                weights[f"{name}:{i}:up"] = weight
            else:
                # 如果是下 (lora_down) 层，存储权重
                weights[f"{name}:{i}:down"] = weight

    # 遍历嵌入字典中的每个 token
    for token, tensor in embeds.items():
        # 将嵌入标志存储在元数据字典中
        metadata[token] = EMBED_FLAG
        # 将嵌入张量添加到权重字典中
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    # 调用 safe_save 函数，将权重和元数据保存到指定的 safetensor 文件中
    safe_save(weights, outpath, metadata)


def convert_loras_to_safeloras(
    modelmap: Dict[str, Tuple[str, Set[str], int]] = {},
    outpath="./lora.safetensors",
):
    """
    将多个 PyTorch.pt 文件中的 LoRA 转换为单个 safetensor 文件。

    参数:
        modelmap (Dict[str, Tuple[str, Set[str], int]], 可选): 模型映射字典，键为模块名称，值为 (PyTorch模型路径, 目标替换模块集合, LoRA秩) 的元组，默认为空字典。
        outpath (str, 可选): 输出文件路径，默认为 "./lora.safetensors"。
    """
    # 调用 convert_loras_to_safeloras_with_embeds 函数，嵌入字典为空字典
    convert_loras_to_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def parse_safeloras(
    safeloras,
) -> Dict[str, Tuple[List[nn.parameter.Parameter], List[int], List[str]]]:
    """
    将加载的包含一组模块 LoRA 的 safetensor 文件转换为参数和其他信息。

    输出是一个字典，格式为 {
        "模块名称": (
            [权重列表],
            [秩列表],
            目标替换模块集合
        )
    }

    参数:
        safeloras: 已加载的包含 LoRA 的 safetensor 文件。

    返回:
        Dict[str, Tuple[List[nn.parameter.Parameter], List[int], List[str]]]: 包含 LoRA 参数、秩和目标替换模块集合的字典。
    """
    # 初始化 LoRA 字典
    loras = {}
    # 获取 safetensor 文件的元数据
    metadata = safeloras.metadata()

    # 定义一个 lambda 函数，用于提取模块名称
    get_name = lambda k: k.split(":")[0]

    # 获取所有键并按键排序
    keys = list(safeloras.keys())
    keys.sort(key=get_name)

    # 按模块名称对键进行分组
    for name, module_keys in groupby(keys, get_name):
        # 获取模块的元数据
        info = metadata.get(name)

        # 如果没有元数据，则抛出 ValueError 异常
        if not info:
            raise ValueError(
                f"Tensor {name} has no metadata - is this a Lora safetensor?"
            )

        # 如果元数据是嵌入标志，则跳过（假设这是文本嵌入）
        if info == EMBED_FLAG:
            continue

        # 处理 LoRA
        # 解析目标替换模块集合
        target = json.loads(info)

        # 初始化秩列表和权重列表
        module_keys = list(module_keys)
        ranks = [4] * (len(module_keys) // 2)
        weights = [None] * len(module_keys)

        # 遍历模块键
        for key in module_keys:
            # 分割键以提取模块名称、索引和方向（up 或 down）
            _, idx, direction = key.split(":")
            idx = int(idx)

            # 获取秩并存储在秩列表中
            ranks[idx] = int(metadata[f"{name}:{idx}:rank"])

            # 计算索引并存储权重
            idx = idx * 2 + (1 if direction == "down" else 0)
            weights[idx] = nn.parameter.Parameter(safeloras.get_tensor(key))

        # 将模块名称、权重列表、秩列表和目标替换模块集合添加到 LoRA 字典中
        loras[name] = (weights, ranks, target)

    return loras


def parse_safeloras_embeds(
    safeloras,
) -> Dict[str, torch.Tensor]:
    """
    将加载的包含文本嵌入的 safetensor 文件转换为字典，格式为 embed_token: Tensor。

    参数:
        safeloras: 已加载的包含文本嵌入的 safetensor 文件。

    返回:
        Dict[str, torch.Tensor]: 包含嵌入 token 和对应张量的字典。
    """
    # 初始化嵌入字典
    embeds = {}
    # 获取 safetensor 文件的元数据
    metadata = safeloras.metadata()

    # 遍历所有键
    for key in safeloras.keys():
        # 获取元数据
        meta = metadata.get(key)
        # 如果元数据不存在或不是嵌入标志，则跳过
        if not meta or meta != EMBED_FLAG:
            continue
        
        # 将嵌入张量添加到嵌入字典中
        embeds[key] = safeloras.get_tensor(key)

    return embeds


def load_safeloras(path, device="cpu"):
    """
    从指定的 safetensor 文件中加载 LoRA，并解析为参数和其他信息。

    参数:
        path (str): 包含 LoRA 的 safetensor 文件路径。
        device (str, 可选): 设备类型，默认为 "cpu"。

    返回:
        Dict[str, Tuple[List[nn.parameter.Parameter], List[int], List[str]]]: 
            一个字典，键为模块名称，值为包含权重列表、秩列表和目标替换模块集合的元组。
    """
    # 使用 safe_open 函数打开 safetensor 文件，并指定框架为 "pt"（PyTorch）和设备类型
    safeloras = safe_open(path, framework="pt", device=device)
    # 解析 safetensor 文件中的 LoRA 信息
    return parse_safeloras(safeloras)


def load_safeloras_embeds(path, device="cpu"):
    """
    从指定的 safetensor 文件中加载文本嵌入，并解析为字典。

    参数:
        path (str): 包含嵌入的 safetensor 文件路径。
        device (str, 可选): 设备类型，默认为 "cpu"。

    返回:
        Dict[str, torch.Tensor]: 一个字典，键为嵌入 token，值为对应的张量。
    """
    # 使用 safe_open 函数打开 safetensor 文件，并指定框架为 "pt"（PyTorch）和设备类型
    safeloras = safe_open(path, framework="pt", device=device)
    # 解析 safetensor 文件中的嵌入信息
    return parse_safeloras_embeds(safeloras)


def load_safeloras_both(path, device="cpu"):
    """
    从指定的 safetensor 文件中同时加载 LoRA 和文本嵌入，并解析为相应的数据结构。

    参数:
        path (str): 包含 LoRA 和嵌入的 safetensor 文件路径。
        device (str, 可选): 设备类型，默认为 "cpu"。

    返回:
        Tuple[Dict[str, Tuple[List[nn.parameter.Parameter], List[int], List[str]]], Dict[str, torch.Tensor]]:
            一个元组，包含两个字典：
                - 第一个字典包含 LoRA 信息，键为模块名称，值为包含权重列表、秩列表和目标替换模块集合的元组。
                - 第二个字典包含嵌入信息，键为嵌入 token，值为对应的张量。
    """
    # 使用 safe_open 函数打开 safetensor 文件，并指定框架为 "pt"（PyTorch）和设备类型
    safeloras = safe_open(path, framework="pt", device=device)
    # 解析 safetensor 文件中的 LoRA 和嵌入信息
    return parse_safeloras(safeloras), parse_safeloras_embeds(safeloras)


def collapse_lora(model, alpha=1.0):
    """
    将 LoRA 层合并到原始模型中。

    参数:
        model (nn.Module): 需要合并 LoRA 的 PyTorch 模型。
        alpha (float, 可选): 合并 LoRA 时的缩放因子，默认为1.0。
    """
    # 遍历模型中所有需要替换的目标模块，查找已注入的 LoRA 层
    for _module, name, _child_module in _find_modules(
        model,
        UNET_EXTENDED_TARGET_REPLACE | TEXT_ENCODER_EXTENDED_TARGET_REPLACE, # 目标替换模块集合为 UNET 和文本编码器的扩展集合
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        # 如果是 LoraInjectedLinear 层
        if isinstance(_child_module, LoraInjectedLinear):
            print("Collapsing Lin Lora in", name)
            # 将原始线性层的权重与 LoRA 调整部分合并
            _child_module.linear.weight = nn.Parameter(
                _child_module.linear.weight.data
                + alpha
                * (
                    _child_module.lora_up.weight.data
                    @ _child_module.lora_down.weight.data
                )
                .type(_child_module.linear.weight.dtype)
                .to(_child_module.linear.weight.device)
            )
        # 如果是 LoraInjectedConv2d 层
        else:
            print("Collapsing Conv Lora in", name)
            # 将原始卷积层的权重与 LoRA 调整部分合并
            _child_module.conv.weight = nn.Parameter(
                _child_module.conv.weight.data
                + alpha
                * (
                    _child_module.lora_up.weight.data.flatten(start_dim=1)
                    @ _child_module.lora_down.weight.data.flatten(start_dim=1)
                )
                .reshape(_child_module.conv.weight.data.shape)
                .type(_child_module.conv.weight.dtype)
                .to(_child_module.conv.weight.device)
            )


def monkeypatch_or_replace_lora(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    r: Union[int, List[int]] = 4,
):
    """
    对模型进行补丁（monkey patch）或替换 LoRA 层。

    参数:
        model (nn.Module): 需要注入或替换 LoRA 的 PyTorch 模型。
        loras (List[torch.Tensor]): LoRA 权重列表，包含上 (lora_up) 和下 (lora_down) 层的权重。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 DEFAULT_TARGET_REPLACE。
        r (Union[int, List[int]], 可选): LoRA 的秩（rank），可以是单个整数或整数列表，默认为4。
    """
    # 遍历模型中所有需要替换的目标模块，查找线性层和已注入的 LoRA 层
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, LoraInjectedLinear]
    ):
        # 确定源模块是原始线性层还是已注入的 LoRA 层
        _source = (
            _child_module.linear
            if isinstance(_child_module, LoraInjectedLinear)
            else _child_module
        )

        # 获取权重和偏置
        weight = _source.weight
        bias = _source.bias

        # 创建 LoraInjectedLinear 实例
        _tmp = LoraInjectedLinear(
            _source.in_features,
            _source.out_features,
            _source.bias is not None,
            r=r.pop(0) if isinstance(r, list) else r, # 如果 r 是列表，则弹出第一个元素作为当前 LoRA 的秩
        )

        # 将原始线性层的权重赋值给新的 LoRA 线性层
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # 替换模型中的原始模块为新的 LoRA 模块
        _module._modules[name] = _tmp

        # 获取上 (lora_up) 和下 (lora_down) 层的权重
        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        # 将上 (lora_up) 和下 (lora_down) 层的权重赋值给新的 LoRA 模块
        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        # 将 LoRA 模块移动到与权重相同的设备
        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_lora_extended(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    r: Union[int, List[int]] = 4,
):
    """
    对模型进行补丁（monkey patch）或替换 LoRA 层，支持线性层和卷积层。

    参数:
        model (nn.Module): 需要注入或替换 LoRA 的 PyTorch 模型。
        loras (List[torch.Tensor]): LoRA 权重列表，包含上 (lora_up) 和下 (lora_down) 层的权重。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 DEFAULT_TARGET_REPLACE。
        r (Union[int, List[int]], 可选): LoRA 的秩（rank），可以是单个整数或整数列表，默认为4。
    """
    # 遍历模型中所有需要替换的目标模块，查找线性层、已注入的 LoRA 层、卷积层和已注入的卷积层
    for _module, name, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[nn.Linear, LoraInjectedLinear, nn.Conv2d, LoraInjectedConv2d],
    ):
        # 如果是线性层或已注入的线性层
        if (_child_module.__class__ == nn.Linear) or (
            _child_module.__class__ == LoraInjectedLinear
        ):
            # 如果当前 LoRA 权重列表的第一个元素的形状不是二维的，则跳过
            if len(loras[0].shape) != 2:
                continue
            
            # 确定源模块是原始线性层还是已注入的 LoRA 层
            _source = (
                _child_module.linear
                if isinstance(_child_module, LoraInjectedLinear)
                else _child_module
            )

            weight = _source.weight
            bias = _source.bias

            # 创建 LoraInjectedLinear 实例
            _tmp = LoraInjectedLinear(
                _source.in_features,
                _source.out_features,
                _source.bias is not None,
                r=r.pop(0) if isinstance(r, list) else r,
            )
            # 将原始线性层的权重赋值给新的 LoRA 线性层
            _tmp.linear.weight = weight

            if bias is not None:
                _tmp.linear.bias = bias

        # 如果是卷积层或已注入的卷积层
        elif (_child_module.__class__ == nn.Conv2d) or (
            _child_module.__class__ == LoraInjectedConv2d
        ):
            # 如果当前 LoRA 权重列表的第一个元素的形状不是四维的，则跳过
            if len(loras[0].shape) != 4:
                continue

            # 确定源模块是原始卷积层还是已注入的 LoRA 层
            _source = (
                _child_module.conv
                if isinstance(_child_module, LoraInjectedConv2d)
                else _child_module
            )

            weight = _source.weight
            bias = _source.bias

            # 创建 LoraInjectedConv2d 实例
            _tmp = LoraInjectedConv2d(
                _source.in_channels,
                _source.out_channels,
                _source.kernel_size,
                _source.stride,
                _source.padding,
                _source.dilation,
                _source.groups,
                _source.bias is not None,
                r=r.pop(0) if isinstance(r, list) else r,
            )

            # 将原始卷积层的权重赋值给新的 LoRA 卷积层
            _tmp.conv.weight = weight

            if bias is not None:
                _tmp.conv.bias = bias

        # 替换模型中的原始模块为新的 LoRA 模块
        _module._modules[name] = _tmp

        # 获取上 (lora_up) 和下 (lora_down) 层的权重
        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        # 将上 (lora_up) 和下 (lora_down) 层的权重赋值给新的 LoRA 模块
        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )

        # 将 LoRA 模块移动到与权重相同的设备
        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_safeloras(models, safeloras):
    """
    使用 safetensor 文件中的 LoRA 权重对多个模型进行补丁（monkey patch）或替换。

    参数:
        models: 包含多个模型的集合，通常是一个包含模型名称和模型对象的字典或类似结构。
        safeloras: 包含 LoRA 权重的 safetensor 文件或类似对象。
    """
    # 解析 safetensor 文件中的 LoRA 信息
    loras = parse_safeloras(safeloras)

    # 遍历每个 LoRA 模块的信息
    for name, (lora, ranks, target) in loras.items():
        # 从模型集合中获取对应的模型对象
        model = getattr(models, name, None)
        # 如果没有找到对应的模型，则打印提示信息并跳过
        if not model:
            print(f"No model provided for {name}, contained in Lora")
            continue
        # 对模型进行补丁或替换 LoRA 层，支持线性层和卷积层
        monkeypatch_or_replace_lora_extended(model, lora, target, ranks)


def monkeypatch_remove_lora(model):
    """
    从模型中移除所有注入的 LoRA 层，将模型恢复到原始状态。

    参数:
        model (nn.Module): 需要移除 LoRA 的 PyTorch 模型。
    """
    # 遍历模型中所有已注入的 LoRA 层
    for _module, name, _child_module in _find_modules(
        model, search_class=[LoraInjectedLinear, LoraInjectedConv2d]
    ):
        # 如果是 LoraInjectedLinear 层
        if isinstance(_child_module, LoraInjectedLinear):
            # 获取原始线性层的权重和偏置
            _source = _child_module.linear
            weight, bias = _source.weight, _source.bias

            # 创建一个新的线性层，参数与原始线性层相同
            _tmp = nn.Linear(
                _source.in_features, _source.out_features, bias is not None
            )

            # 将原始线性层的权重和偏置赋值给新的线性层
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias

        # 如果是 LoraInjectedConv2d 层
        else:
            # 获取原始卷积层的权重和偏置
            _source = _child_module.conv
            weight, bias = _source.weight, _source.bias

            # 创建一个新的卷积层，参数与原始卷积层相同
            _tmp = nn.Conv2d(
                in_channels=_source.in_channels,
                out_channels=_source.out_channels,
                kernel_size=_source.kernel_size,
                stride=_source.stride,
                padding=_source.padding,
                dilation=_source.dilation,
                groups=_source.groups,
                bias=bias is not None,
            )

            # 将原始卷积层的权重和偏置赋值给新的卷积层
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias

        # 将新的线性层或卷积层替换回原始模块
        _module._modules[name] = _tmp


def monkeypatch_add_lora(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    """
    向模型中添加 LoRA 层，并结合原始权重和 LoRA 权重。

    参数:
        model (nn.Module): 需要添加 LoRA 的 PyTorch 模型。
        loras (List[torch.Tensor]): LoRA 权重列表，包含上 (lora_up) 和下 (lora_down) 层的权重。
        target_replace_module (Set[str], 可选): 需要替换的目标模块名称集合，默认为 DEFAULT_TARGET_REPLACE。
        alpha (float, 可选): LoRA 权重的缩放因子 alpha，默认为1.0。
        beta (float, 可选): 原始权重的缩放因子 beta，默认为1.0。
    """
    # 遍历模型中所有需要替换的目标模块，查找已注入的 LoRA 层
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedLinear]
    ):
        # 获取原始线性层的权重
        weight = _child_module.linear.weight

        # 获取上 (lora_up) 和下 (lora_down) 层的权重
        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        # 更新 LoRA 层的上 (lora_up) 和下 (lora_down) 层的权重，结合原始权重和 LoRA 权重
        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_up.weight.to(weight.device) * beta
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_down.weight.to(weight.device) * beta
        )

        # 将 LoRA 模块移动到与权重相同的设备
        _module._modules[name].to(weight.device)


def tune_lora_scale(model, alpha: float = 1.0):
    """
    调整模型中所有 LoRA 层的缩放因子。

    参数:
        model (nn.Module): 需要调整 LoRA 缩放因子的模型。
        alpha (float, 可选): 新的缩放因子，默认为1.0。
    """
    # 遍历模型中的每个模块
    for _module in model.modules():
        # 如果模块是 LoRA 注入的线性层或卷积层，则调整其缩放因子
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            _module.scale = alpha


def set_lora_diag(model, diag: torch.Tensor):
    """
    设置模型中所有 LoRA 层的选择器为对角矩阵。

    参数:
        model (nn.Module): 需要设置 LoRA 选择器的模型。
        diag (torch.Tensor): 对角矩阵张量，用于设置选择器。
    """
    # 遍历模型中的每个模块
    for _module in model.modules():
        # 如果模块是 LoRA 注入的线性层或卷积层，则设置其选择器为对角矩阵
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            _module.set_selector_from_diag(diag)


def _text_lora_path(path: str) -> str:
    """
    生成文本编码器 LoRA 文件的路径。

    参数:
        path (str): 原始文件路径。

    返回:
        str: 生成的文本编码器 LoRA 文件路径。
    """
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str) -> str:
    """
    生成文本嵌入 LoRA 文件的路径。

    参数:
        path (str): 原始文件路径。

    返回:
        str: 生成的文本嵌入 LoRA 文件路径。
    """
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["ti", "pt"])


def apply_learned_embed_in_clip(
    learned_embeds,
    text_encoder,
    tokenizer,
    token: Optional[Union[str, List[str]]] = None,
    idempotent=False,
):
    """
    在 CLIP 文本编码器中应用学习到的嵌入。

    参数:
        learned_embeds (Dict[str, torch.Tensor]): 包含学习到的嵌入的字典，键为 token，值为对应的嵌入张量。
        text_encoder (nn.Module): CLIP 文本编码器模型。
        tokenizer (transformers.PreTrainedTokenizer): 分词器，用于处理文本。
        token (Optional[Union[str, List[str]]], 可选): 要训练的 token，可以是单个字符串或字符串列表，默认为 None。
        idempotent (bool, 可选): 是否为幂等操作，默认为 False。

    返回:
        List[str]: 训练过的 token 列表。
    """
    # 如果 token 是字符串，则转换为列表
    if isinstance(token, str):
        trained_tokens = [token]
    # 如果 token 是列表，则确保学习到的嵌入数量与 token 数量一致
    elif isinstance(token, list):
        assert len(learned_embeds.keys()) == len(
            token
        ), "The number of tokens and the number of embeds should be the same"
        trained_tokens = token
    # 如果未指定 token，则使用学习到的嵌入中的所有 token
    else:
        trained_tokens = list(learned_embeds.keys())

    # 遍历每个训练过的 token
    for token in trained_tokens:
        print(token)
        embeds = learned_embeds[token]

        # 将嵌入张量转换为文本编码器的权重数据类型
        dtype = text_encoder.get_input_embeddings().weight.dtype
        # 向分词器中添加新的 token
        num_added_tokens = tokenizer.add_tokens(token)

        i = 1
        # 如果不是幂等操作，则处理已经存在的 token
        if not idempotent:
            while num_added_tokens == 0:
                print(f"The tokenizer already contains the token {token}.")
                token = f"{token[:-1]}-{i}>"
                print(f"Attempting to add the token {token}.")
                num_added_tokens = tokenizer.add_tokens(token)
                i += 1
        # 如果是幂等操作且 token 已存在，则替换嵌入
        elif num_added_tokens == 0 and idempotent:
            print(f"The tokenizer already contains the token {token}.")
            print(f"Replacing {token} embedding.")

        # 调整文本编码器的 token 嵌入大小
        text_encoder.resize_token_embeddings(len(tokenizer))

        # 获取 token 的 ID 并分配嵌入
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


def load_learned_embed_in_clip(
    learned_embeds_path,
    text_encoder,
    tokenizer,
    token: Optional[Union[str, List[str]]] = None,
    idempotent=False,
):
    """
    从文件中加载学习到的嵌入并在 CLIP 文本编码器中应用它们。

    参数:
        learned_embeds_path (str): 学习到的嵌入文件路径（.pt 文件）。
        text_encoder (nn.Module): CLIP 文本编码器模型。
        tokenizer (transformers.PreTrainedTokenizer): 分词器，用于处理文本。
        token (Optional[Union[str, List[str]]], 可选): 要加载的 token，可以是单个字符串或字符串列表，默认为 None。
        idempotent (bool, 可选): 是否为幂等操作，默认为 False。
    """
    # 从指定路径加载学习到的嵌入
    learned_embeds = torch.load(learned_embeds_path)
    # 在 CLIP 文本编码器中应用学习到的嵌入
    apply_learned_embed_in_clip(
        learned_embeds, text_encoder, tokenizer, token, idempotent
    )


def patch_pipe(
    pipe,
    maybe_unet_path,
    token: Optional[str] = None,
    r: int = 4,
    patch_unet=True,
    patch_text=True,
    patch_ti=True,
    idempotent_token=True,
    unet_target_replace_module=DEFAULT_TARGET_REPLACE,
    text_target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
):
    """
    对管道模型（包含 UNet 和文本编码器）进行 LoRA 修补。

    参数:
        pipe: 包含 UNet 和文本编码器的管道模型。
        maybe_unet_path (str): 可能包含 UNet LoRA 权重的文件路径。
        token (Optional[str], 可选): 要训练的 token，默认为 None。
        r (int, 可选): LoRA 的秩（rank），默认为4。
        patch_unet (bool, 可选): 是否修补 UNet，默认为 True。
        patch_text (bool, 可选): 是否修补文本编码器，默认为 True。
        patch_ti (bool, 可选): 是否修补 token 输入，默认为 True。
        idempotent_token (bool, 可选): 是否为幂等 token 操作，默认为 True。
        unet_target_replace_module (Set[str], 可选): UNet 替换的目标模块集合，默认为 DEFAULT_TARGET_REPLACE。
        text_target_replace_module (Set[str], 可选): 文本编码器替换的目标模块集合，默认为 TEXT_ENCODER_DEFAULT_TARGET_REPLACE。
    """
    # 检查文件路径是否以 ".pt" 结尾
    if maybe_unet_path.endswith(".pt"):
        # 如果文件路径以 ".ti.pt" 结尾，则生成 UNet LoRA 权重文件路径
        if maybe_unet_path.endswith(".ti.pt"):
            unet_path = maybe_unet_path[:-6] + ".pt"
        # 如果文件路径以 ".text_encoder.pt" 结尾，则生成 UNet LoRA 权重文件路径
        elif maybe_unet_path.endswith(".text_encoder.pt"):
            unet_path = maybe_unet_path[:-16] + ".pt"
        else:
            unet_path = maybe_unet_path

        # 生成 token 输入 LoRA 权重文件路径和文本编码器 LoRA 权重文件路径
        ti_path = _ti_lora_path(unet_path)
        text_path = _text_lora_path(unet_path)

        # 如果需要修补 UNet，则加载 UNet LoRA 权重并应用
        if patch_unet:
            print("LoRA : Patching Unet")
            monkeypatch_or_replace_lora(
                pipe.unet,
                torch.load(unet_path),
                r=r,
                target_replace_module=unet_target_replace_module,
            )

        # 如果需要修补文本编码器，则加载文本编码器 LoRA 权重并应用
        if patch_text:
            print("LoRA : Patching text encoder")
            monkeypatch_or_replace_lora(
                pipe.text_encoder,
                torch.load(text_path),
                target_replace_module=text_target_replace_module,
                r=r,
            )
        
        # 如果需要修补 token 输入，则加载 token 输入 LoRA 权重并应用
        if patch_ti:
            print("LoRA : Patching token input")
            token = load_learned_embed_in_clip(
                ti_path,
                pipe.text_encoder,
                pipe.tokenizer,
                token=token,
                idempotent=idempotent_token,
            )

    # 如果文件路径以 ".safetensors" 结尾，则处理 safetensor 文件
    elif maybe_unet_path.endswith(".safetensors"):
        # 打开 safetensor 文件
        safeloras = safe_open(maybe_unet_path, framework="pt", device="cpu")
        # 对管道模型应用 safetensor 文件中的 LoRA 权重
        monkeypatch_or_replace_safeloras(pipe, safeloras)
        # 解析嵌入字典
        tok_dict = parse_safeloras_embeds(safeloras)
        # 如果需要修补 token 输入，则应用学习到的嵌入
        if patch_ti:
            apply_learned_embed_in_clip(
                tok_dict,
                pipe.text_encoder,
                pipe.tokenizer,
                token=token,
                idempotent=idempotent_token,
            )
        # 返回嵌入字典
        return tok_dict


# 禁用梯度计算，节省显存和加速计算
@torch.no_grad()
def inspect_lora(model):
    """
    检查模型中所有注入的 LoRA 层的权重分布。

    参数:
        model (nn.Module): 需要检查的模型。

    返回:
        Dict[str, List[float]]: 一个字典，键为模块名称，值为包含权重分布距离的列表。
    """
    moved = {}

    # 遍历模型中的每个模块
    for name, _module in model.named_modules():
        # 如果模块是 LoRA 注入的线性层或卷积层
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            # 克隆 LoRA 上层和下层的权重
            ups = _module.lora_up.weight.data.clone()
            downs = _module.lora_down.weight.data.clone()

            # 计算 LoRA 调整部分的权重
            wght: torch.Tensor = ups.flatten(1) @ downs.flatten(1)

            # 计算权重分布的距离（平均绝对值）
            dist = wght.flatten().abs().mean().item()
            # 将距离添加到对应的模块名称下
            if name in moved:
                moved[name].append(dist)
            else:
                moved[name] = [dist]
    # 返回包含权重分布距离的字典
    return moved


def save_all(
    unet,
    text_encoder,
    save_path,
    placeholder_token_ids=None,
    placeholder_tokens=None,
    save_lora=True,
    save_ti=True,
    target_replace_module_text=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    target_replace_module_unet=DEFAULT_TARGET_REPLACE,
    safe_form=True,
):
    """
    保存 UNet 和文本编码器的 LoRA 权重以及 token 嵌入。

    参数:
        unet (nn.Module): UNet 模型。
        text_encoder (nn.Module): 文本编码器模型。
        save_path (str): 保存路径。
        placeholder_token_ids (Optional[List[int]], 可选): 占位符 token ID 列表，默认为 None。
        placeholder_tokens (Optional[List[str]], 可选): 占位符 token 列表，默认为 None。
        save_lora (bool, 可选): 是否保存 LoRA 权重，默认为 True。
        save_ti (bool, 可选): 是否保存 token 嵌入，默认为 True。
        target_replace_module_text (Set[str], 可选): 文本编码器替换的目标模块集合，默认为 TEXT_ENCODER_DEFAULT_TARGET_REPLACE。
        target_replace_module_unet (Set[str], 可选): UNet 替换的目标模块集合，默认为 DEFAULT_TARGET_REPLACE。
        safe_form (bool, 可选): 是否以 safetensor 格式保存，默认为 True。
    """
    if not safe_form:
        # 保存 token 嵌入
        if save_ti:
            ti_path = _ti_lora_path(save_path)
            learned_embeds_dict = {}
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(
                    f"Current Learned Embeddings for {tok}:, id {tok_id} ",
                    learned_embeds[:4],
                )
                learned_embeds_dict[tok] = learned_embeds.detach().cpu()

            torch.save(learned_embeds_dict, ti_path)
            print("Ti saved to ", ti_path)

        # 保存 LoRA 权重
        if save_lora:

            save_lora_weight(
                unet, save_path, target_replace_module=target_replace_module_unet
            )
            print("Unet saved to ", save_path)

            save_lora_weight(
                text_encoder,
                _text_lora_path(save_path),
                target_replace_module=target_replace_module_text,
            )
            print("Text Encoder saved to ", _text_lora_path(save_path))

    else:
        assert save_path.endswith(
            ".safetensors"
        ), f"Save path : {save_path} should end with .safetensors"

        loras = {}
        embeds = {}

        # 如果需要保存 LoRA
        if save_lora:

            loras["unet"] = (unet, target_replace_module_unet)
            loras["text_encoder"] = (text_encoder, target_replace_module_text)

        # 如果需要保存 token 嵌入
        if save_ti:
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(
                    f"Current Learned Embeddings for {tok}:, id {tok_id} ",
                    learned_embeds[:4],
                )
                embeds[tok] = learned_embeds.detach().cpu()

        # 使用 safetensor 格式保存 LoRA 和嵌入
        save_safeloras_with_embeds(loras, embeds, save_path)
