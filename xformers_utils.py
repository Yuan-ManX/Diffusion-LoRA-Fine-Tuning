import functools
import torch
from diffusers.models.attention import BasicTransformerBlock
from diffusers.utils.import_utils import is_xformers_available

from lora import LoraInjectedLinear

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


@functools.cache
def test_xformers_backwards(size):
    """
    测试xformers的memory_efficient_attention在给定尺寸下的反向传播是否成功。

    参数:
    - size (int): 输入张量的尺寸，用于创建随机张量。

    返回:
    - bool: 如果反向传播成功，则返回True，否则返回False。
    """
    # 启用梯度计算
    @torch.enable_grad()
    def _grad(size):
        # 创建随机张量q, k, v，并移动到GPU（如果可用）
        q = torch.randn((1, 4, size), device="cuda")
        k = torch.randn((1, 4, size), device="cuda")
        v = torch.randn((1, 4, size), device="cuda")

        # 分离张量并启用梯度计算
        q = q.detach().requires_grad_()
        k = k.detach().requires_grad_()
        v = v.detach().requires_grad_()

        # 使用xformers的memory_efficient_attention进行注意力计算
        out = xformers.ops.memory_efficient_attention(q, k, v)
        # 计算损失，这里是对输出的某个维度求和并取均值
        loss = out.sum(2).mean(0).sum()

        # 计算损失的梯度相对于v的梯度
        return torch.autograd.grad(loss, v)

    try:
        # 执行梯度计算
        _grad(size)
        print(size, "pass")
        return True
    except Exception as e:
        print(size, "fail")
        return False


def set_use_memory_efficient_attention_xformers(
    module: torch.nn.Module, valid: bool
) -> None:
    """
    设置模块及其子模块是否使用xformers的memory_efficient_attention。

    参数:
    - module (torch.nn.Module): 要设置的PyTorch模块。
    - valid (bool): 如果为True，则启用xformers的memory_efficient_attention；否则，禁用。

    返回:
    - None
    """
    def fn_test_dim_head(module: torch.nn.Module):
        """
        测试给定模块的dim_head是否支持xformers的memory_efficient_attention。

        参数:
        - module (torch.nn.Module): 要测试的模块。

        返回:
        - None
        """
        if isinstance(module, BasicTransformerBlock):
            # 获取dim_head，假设可以通过attn1.to_v的输出特征数除以注意力头的数量得到
            source = module.attn1.to_v
            if isinstance(source, LoraInjectedLinear):
                # 如果是LoraInjectedLinear，则获取原始的线性层
                source = source.linear

            # 计算dim_head
            dim_head = source.out_features // module.attn1.heads

            # 测试xformers的反向传播是否成功
            result = test_xformers_backwards(dim_head)

            # 如果dim_head大于某个阈值（假设为某个内部常量dim_head_max），则禁用xformers
            if not result:
                module.set_use_memory_efficient_attention_xformers(False)

        # 递归地测试子模块
        for child in module.children():
            fn_test_dim_head(child)

    if not is_xformers_available() and valid:
        # 如果xformers不可用，则跳过
        print("XFormers is not available. Skipping.")
        return

    # 设置模块是否使用xformers
    module.set_use_memory_efficient_attention_xformers(valid)

    if valid:
        # 如果启用xformers，则测试dim_head
        fn_test_dim_head(module)
