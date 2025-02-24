from typing import List
import torch
from safetensors import safe_open
from diffusers import StableDiffusionPipeline

from lora import monkeypatch_or_replace_safeloras, apply_learned_embed_in_clip, set_lora_diag, parse_safeloras_embeds


def lora_join(lora_safetenors: list):
    """
    将多个 LoRA safetensor 文件合并为一个统一的 LoRA 表示。

    参数:
        lora_safetenors (list): 包含多个 LoRA safetensor 文件的列表，每个文件都是一个 safelora 对象。

    返回:
        Tuple[dict, dict, list, list]: 
            - total_tensor (dict): 合并后的张量字典，键为张量名称，值为对应的张量。
            - total_metadata (dict): 合并后的元数据字典，键为元数据键，值为对应的值。
            - ranklist (list): 每个 safetensor 文件的 LoRA 秩（rank）列表。
            - token_size_list (list): 每个 safetensor 文件中嵌入 token 的数量列表。
    """
    # 提取每个 safetensor 文件的元数据，并转换为字典
    metadatas = [dict(safelora.metadata()) for safelora in lora_safetenors]
    # 初始化合并后的元数据字典
    _total_metadata = {}
    total_metadata = {}
    # 初始化合并后的张量字典
    total_tensor = {}
    # 初始化总秩（rank）变量
    total_rank = 0
    # 初始化每个 safetensor 文件的秩列表
    ranklist = []

    # 遍历每个 safetensor 文件的元数据
    for _metadata in metadatas:
        rankset = []
        # 遍历元数据中的每个键值对
        for k, v in _metadata.items():
            # 如果键以 "rank" 结尾，则将其值转换为整数并添加到 rankset 列表中
            if k.endswith("rank"):
                rankset.append(int(v))

        # 检查所有秩是否相同
        assert len(set(rankset)) <= 1, "Rank should be the same per model"
        # 如果没有找到任何秩，则默认秩为0
        if len(rankset) == 0:
            rankset = [0]

        # 将当前 safetensor 文件的秩加到总秩中
        total_rank += rankset[0]
        # 更新总元数据字典
        _total_metadata.update(_metadata)
        # 将当前 safetensor 文件的秩添加到秩列表中
        ranklist.append(rankset[0])

    # 移除与 token 相关的元数据
    for k, v in _total_metadata.items():
        if v != "<embed>":
            total_metadata[k] = v

    # 获取所有 safetensor 文件中的所有张量键
    tensorkeys = set()
    for safelora in lora_safetenors:
        tensorkeys.update(safelora.keys())
    
    # 遍历所有张量键
    for keys in tensorkeys:
        # 如果键以 "text_encoder" 或 "unet" 开头，则处理 LoRA 权重张量
        if keys.startswith("text_encoder") or keys.startswith("unet"):
            # 从每个 safetensor 文件中提取对应的张量
            tensorset = [safelora.get_tensor(keys) for safelora in lora_safetenors]

            # 判断当前张量是否为下层（down）权重
            is_down = keys.endswith("down")

            if is_down:
                # 如果是下层权重，则在第0维进行拼接
                _tensor = torch.cat(tensorset, dim=0)
                # 确保拼接后的张量在第0维的尺寸等于总秩
                assert _tensor.shape[0] == total_rank
            else:
                # 如果是上层（up）权重，则在第1维进行拼接
                _tensor = torch.cat(tensorset, dim=1)
                # 确保拼接后的张量在第1维的尺寸等于总秩
                assert _tensor.shape[1] == total_rank

            # 将拼接后的张量添加到总张量字典中
            total_tensor[keys] = _tensor
            # 更新元数据中的秩信息
            keys_rank = ":".join(keys.split(":")[:-1]) + ":rank"
            total_metadata[keys_rank] = str(total_rank)
    
    # 处理嵌入 token
    token_size_list = []
    for idx, safelora in enumerate(lora_safetenors):
        # 获取当前 safetensor 文件中的所有嵌入 token
        tokens = [k for k, v in safelora.metadata().items() if v == "<embed>"]
        for jdx, token in enumerate(sorted(tokens)):    
            # 将嵌入张量添加到总张量字典中，并重命名 token
            total_tensor[f"<s{idx}-{jdx}>"] = safelora.get_tensor(token)
            # 更新元数据，标识为嵌入
            total_metadata[f"<s{idx}-{jdx}>"] = "<embed>"

            # 打印替换信息
            print(f"Embedding {token} replaced to <s{idx}-{jdx}>")

        # 记录每个 safetensor 文件中的嵌入 token 数量
        token_size_list.append(len(tokens))

    # 返回合并后的张量、元数据、秩列表和嵌入 token 数量列表
    return total_tensor, total_metadata, ranklist, token_size_list


class DummySafeTensorObject:
    """
    一个模拟的 safetensor 对象，用于封装张量和元数据。

    属性:
        tensor (dict): 包含张量的字典，键为张量名称，值为对应的张量。
        _metadata (dict): 包含元数据的字典，键为元数据键，值为对应的值。
    """
    def __init__(self, tensor: dict, metadata):
        """
        初始化 DummySafeTensorObject 实例。

        参数:
            tensor (dict): 包含张量的字典。
            metadata (dict): 包含元数据的字典。
        """
        self.tensor = tensor
        self._metadata = metadata

    def keys(self):
        """
        获取张量字典的键。

        返回:
            dict_keys: 张量字典的键。
        """
        return self.tensor.keys()

    def metadata(self):
        """
        获取元数据字典。

        返回:
            dict: 元数据字典。
        """
        return self._metadata

    def get_tensor(self, key):
        """
        获取指定键的张量。

        参数:
            key (str): 张量键。

        返回:
            torch.Tensor: 对应的张量。
        """
        return self.tensor[key]


class LoRAManager:
    """
    LoRAManager 类用于管理和应用多个 LoRA（低秩适应）权重到 Stable Diffusion 管道模型中。
    它支持加载多个 LoRA 权重文件，合并它们，并调整缩放因子以控制 LoRA 的强度。

    参数:
        lora_paths_list (List[str]): 包含多个 LoRA 权重文件路径的列表。
        pipe (StableDiffusionPipeline): Stable Diffusion 管道模型，用于应用 LoRA 权重。
    """
    def __init__(self, lora_paths_list: List[str], pipe: StableDiffusionPipeline):
        """
        初始化 LoRAManager 实例。

        参数:
            lora_paths_list (List[str]): 包含多个 LoRA 权重文件路径的列表。
            pipe (StableDiffusionPipeline): Stable Diffusion 管道模型，用于应用 LoRA 权重。
        """
        self.lora_paths_list = lora_paths_list
        self.pipe = pipe
        self._setup()

    def _setup(self):
        """
        设置方法，用于加载和合并 LoRA 权重，并将其应用到管道模型中。
        """
        # 加载所有 LoRA safetensor 文件，并存储在列表中
        self._lora_safetenors = [
            safe_open(path, framework="pt", device="cpu")
            for path in self.lora_paths_list
        ]

        # 合并所有 LoRA safetensor 文件，返回总张量、总元数据、秩列表和嵌入 token 大小列表
        (
            total_tensor,
            total_metadata,
            self.ranklist,
            self.token_size_list,
        ) = lora_join(self._lora_safetenors)

        # 使用合并后的张量和元数据创建一个模拟的 safetensor 对象
        self.total_safelora = DummySafeTensorObject(total_tensor, total_metadata)

        # 对管道模型应用合并后的 LoRA 权重
        monkeypatch_or_replace_safeloras(self.pipe, self.total_safelora)
        # 解析嵌入字典
        tok_dict = parse_safeloras_embeds(self.total_safelora)

        # 在 CLIP 文本编码器中应用学习到的嵌入
        apply_learned_embed_in_clip(
            tok_dict,
            self.pipe.text_encoder,
            self.pipe.tokenizer,
            token=None,  # 不指定具体的 token，应用所有嵌入
            idempotent=True,  # 为幂等操作，避免重复添加 token
        )

    def tune(self, scales):
        """
        调整 LoRA 缩放因子，以控制 LoRA 的强度。

        参数:
            scales (List[float]): 每个 LoRA 权重文件的缩放因子列表。

        异常:
            AssertionError: 如果缩放因子列表的长度与秩列表的长度不一致，则抛出此异常。
        """
        # 确保缩放因子列表的长度与秩列表的长度相同
        assert len(scales) == len(
            self.ranklist
        ), "Scale list should be the same length as ranklist"

        # 根据每个 LoRA 权重文件的缩放因子和秩，生成对角矩阵列表
        diags = []
        for scale, rank in zip(scales, self.ranklist):
            diags = diags + [scale] * rank

        # 将对角矩阵列表转换为张量，并应用到 UNet 模型中
        set_lora_diag(self.pipe.unet, torch.tensor(diags))

    def prompt(self, prompt):
        """
        处理提示（prompt），将占位符替换为实际的嵌入 token。

        参数:
            prompt (str): 输入的提示文本，可能包含占位符。

        返回:
            str: 处理后的提示文本，占位符被替换为实际的嵌入 token。

        备注:
            TODO: 根据提示缩放参数重新调整 LoRA 和文本输入。
        """
        if prompt is not None:
            # 遍历每个嵌入 token 大小列表
            for idx, tok_size in enumerate(self.token_size_list):
                # 将占位符 "<1>", "<2>" 等替换为实际的嵌入 token
                prompt = prompt.replace(
                    f"<{idx + 1}>",
                    "".join([f"<s{idx}-{jdx}>" for jdx in range(tok_size)]),
                )
        # TODO: 根据提示缩放参数重新调整 LoRA 和文本输入

        return prompt
