from typing import List, Union
import os
import glob
import math
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline

from lora import patch_pipe, tune_lora_scale, _text_lora_path, _ti_lora_path


# 示例的文本提示列表，用于生成图像或进行文本-图像对齐
EXAMPLE_PROMPTS = [
    "<obj> swimming in a pool",
    "<obj> at a beach with a view of seashore",
    "<obj> in times square",
    "<obj> wearing sunglasses",
    "<obj> in a construction outfit",
    "<obj> playing with a ball",
    "<obj> wearing headphones",
    "<obj> oil painting ghibli inspired",
    "<obj> working on the laptop",
    "<obj> with mountains and sunset in background",
    "Painting of <obj> at a beach by artist claude monet",
    "<obj> digital painting 3d render geometric style",
    "A screaming <obj>",
    "A depressed <obj>",
    "A sleeping <obj>",
    "A sad <obj>",
    "A joyous <obj>",
    "A frowning <obj>",
    "A sculpture of <obj>",
    "<obj> near a pool",
    "<obj> at a beach with a view of seashore",
    "<obj> in a garden",
    "<obj> in grand canyon",
    "<obj> floating in ocean",
    "<obj> and an armchair",
    "A maple tree on the side of <obj>",
    "<obj> and an orange sofa",
    "<obj> with chocolate cake on it",
    "<obj> with a vase of rose flowers on it",
    "A digital illustration of <obj>",
    "Georgia O'Keeffe style <obj> painting",
    "A watercolor painting of <obj> on a beach",
]


def image_grid(_imgs, rows=None, cols=None):
    """
    将一组图像排列成一个网格图像。

    参数:
    - _imgs (list of PIL.Image.Image): 要排列的图像列表。
    - rows (int, 可选): 网格的行数。如果未指定，将根据cols自动计算。
    - cols (int, 可选): 网格的列数。如果未指定，将根据rows自动计算。

    返回:
    - grid (PIL.Image.Image): 排列后的网格图像。
    """
    if rows is None and cols is None:
        # 如果行和列都未指定，则计算一个接近平方根的值作为行和列
        rows = cols = math.ceil(len(_imgs) ** 0.5)

    if rows is None:
        # 如果只指定了列，则计算行数
        rows = math.ceil(len(_imgs) / cols)
    if cols is None:
        # 如果只指定了行，则计算列数
        cols = math.ceil(len(_imgs) / rows)

    # 获取第一张图像的宽度和高度
    w, h = _imgs[0].size
    # 创建一个新的RGB图像作为网格
    grid = Image.new("RGB", size=(cols * w, rows * h))
    # 获取网格的宽度和高度
    grid_w, grid_h = grid.size

    # 将每张图像粘贴到网格的相应位置
    for i, img in enumerate(_imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def text_img_alignment(img_embeds, text_embeds, target_img_embeds):
    """
    计算文本嵌入与图像嵌入之间的对齐度，以及图像嵌入与目标图像嵌入之间的对齐度。

    参数:
    - img_embeds (torch.Tensor): 图像嵌入，形状为 (N, D)。
    - text_embeds (torch.Tensor): 文本嵌入，形状为 (N, D)。
    - target_img_embeds (torch.Tensor): 目标图像嵌入，形状为 (M, D)。

    返回:
    - alignment_dict (dict): 对齐度结果，包括：
        - text_alignment_avg (float): 文本与图像嵌入的平均对齐度。
        - image_alignment_avg (float): 图像与目标图像嵌入的平均对齐度。
        - text_alignment_all (list of float): 每对文本与图像嵌入的对齐度列表。
        - image_alignment_all (list of float): 每对图像与目标图像嵌入的对齐度列表。
    """
    assert img_embeds.shape[0] == text_embeds.shape[0]
    # 计算文本与图像嵌入的点积相似度
    text_img_sim = (img_embeds * text_embeds).sum(dim=-1) / (
        img_embeds.norm(dim=-1) * text_embeds.norm(dim=-1)
    )

    # 归一化图像嵌入
    img_embed_normalized = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

    # 计算平均目标图像嵌入，并重复以匹配图像嵌入的数量
    avg_target_img_embed = (
        (target_img_embeds / target_img_embeds.norm(dim=-1, keepdim=True))
        .mean(dim=0)
        .unsqueeze(0)
        .repeat(img_embeds.shape[0], 1)
    )

    # 计算图像与平均目标图像嵌入的点积相似度
    img_img_sim = (img_embed_normalized * avg_target_img_embed).sum(dim=-1)

    # 返回对齐度结果
    return {
        "text_alignment_avg": text_img_sim.mean().item(),
        "image_alignment_avg": img_img_sim.mean().item(),
        "text_alignment_all": text_img_sim.tolist(),
        "image_alignment_all": img_img_sim.tolist(),
    }


def prepare_clip_model_sets(eval_clip_id: str = "openai/clip-vit-large-patch14"):
    """
    准备CLIP模型的各个组件。

    参数:
    - eval_clip_id (str): CLIP模型的标识符，默认为 'openai/clip-vit-large-patch14'。

    返回:
    - text_model (CLIPTextModelWithProjection): 文本模型。
    - tokenizer (CLIPTokenizer): 分词器。
    - vis_model (CLIPVisionModelWithProjection): 视觉模型。
    - processor (CLIPProcessor): 处理器，用于处理输入数据。
    """
    # 加载文本模型和投影层
    text_model = CLIPTextModelWithProjection.from_pretrained(eval_clip_id)
    # 加载分词器
    tokenizer = CLIPTokenizer.from_pretrained(eval_clip_id)
    # 加载视觉模型和投影层
    vis_model = CLIPVisionModelWithProjection.from_pretrained(eval_clip_id)
    # 加载处理器，用于处理图像和文本
    processor = CLIPProcessor.from_pretrained(eval_clip_id)

    return text_model, tokenizer, vis_model, processor


def evaluate_pipe(
    pipe,
    target_images: List[Image.Image],
    class_token: str = "",
    learnt_token: str = "",
    guidance_scale: float = 5.0,
    seed=0,
    clip_model_sets=None,
    eval_clip_id: str = "openai/clip-vit-large-patch14",
    n_test: int = 10,
    n_step: int = 50,
):
    """
    评估文本到图像生成管道的性能。

    参数:
    - pipe: Stable Diffusion管道对象，用于生成图像。
    - target_images (List[Image.Image]): 目标图像列表，用于比较。
    - class_token (str): 类别标记，用于替换学习到的标记。
    - learnt_token (str): 学习到的标记，用于文本提示中。
    - guidance_scale (float): 指导尺度，控制生成图像的多样性。
    - seed (int): 随机种子，用于结果可重复性。
    - clip_model_sets: CLIP模型的各个组件（文本模型、分词器、视觉模型、处理器）。如果为None，则使用默认的CLIP模型。
    - eval_clip_id (str): 用于评估的CLIP模型标识符，默认为 'openai/clip-vit-large-patch14'。
    - n_test (int): 要评估的提示数量，默认为10。
    - n_step (int): 生成图像的推理步数，默认为50。

    返回:
    - alignment_results (dict): 对齐度结果，包括文本-图像对齐和图像-目标图像对齐的平均值和所有值。
    """
    # 如果提供了CLIP模型组件，则使用它们；否则，使用默认的CLIP模型进行准备
    if clip_model_sets is not None:
        text_model, tokenizer, vis_model, processor = clip_model_sets
    else:
        text_model, tokenizer, vis_model, processor = prepare_clip_model_sets(
            eval_clip_id
        )

    images = []          # 存储生成的图像
    img_embeds = []      # 存储图像嵌入
    text_embeds = []     # 存储文本嵌入

    # 遍历前n_test个提示
    for prompt in EXAMPLE_PROMPTS[:n_test]:
        # 替换占位符 <obj> 为学习到的标记
        prompt = prompt.replace("<obj>", learnt_token)
        # 设置随机种子以确保结果可重复
        torch.manual_seed(seed)
        # 使用自动混合精度加速推理
        with torch.autocast("cuda"):
            # 生成图像
            img = pipe(
                prompt, num_inference_steps=n_step, guidance_scale=guidance_scale
            ).images[0]
        images.append(img)

        # 获取图像嵌入
        # 处理图像并转换为张量
        inputs = processor(images=img, return_tensors="pt")
        # 获取图像嵌入
        img_embed = vis_model(**inputs).image_embeds
        img_embeds.append(img_embed)

        # 替换学习到的标记为类别标记，用于文本嵌入计算
        prompt = prompt.replace(learnt_token, class_token)
        # 处理文本提示
        # 分词并转换为张量
        inputs = tokenizer([prompt], padding=True, return_tensors="pt")
        # 获取文本模型输出
        outputs = text_model(**inputs)
        # 获取文本嵌入
        text_embed = outputs.text_embeds
        text_embeds.append(text_embed)

    # 获取目标图像的嵌入
    inputs = processor(images=target_images, return_tensors="pt")
    target_img_embeds = vis_model(**inputs).image_embeds

    # 将所有图像嵌入和文本嵌入连接成一个张量
    img_embeds = torch.cat(img_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)

    # 计算对齐度
    return text_img_alignment(img_embeds, text_embeds, target_img_embeds)


def visualize_progress(
    path_alls: Union[str, List[str]],
    prompt: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    device="cuda:0",
    patch_unet=True,
    patch_text=True,
    patch_ti=True,
    unet_scale=1.0,
    text_sclae=1.0,
    num_inference_steps=50,
    guidance_scale=5.0,
    offset: int = 0,
    limit: int = 10,
    seed: int = 0,
):
    """
    可视化模型训练的进度，通过生成图像并展示。

    参数:
    - path_alls (str 或 List[str]): 模型检查点文件的路径或路径列表。
    - prompt (str): 用于生成图像的文本提示。
    - model_id (str): 预训练的Stable Diffusion模型标识符，默认为 'runwayml/stable-diffusion-v1-5'。
    - device (str): 计算设备，默认为 'cuda:0'。
    - patch_unet (bool): 是否补丁UNet模型。
    - patch_text (bool): 是否补丁文本编码器。
    - patch_ti (bool): 是否补丁文本反转模块。
    - unet_scale (float): UNet模型的缩放因子，默认为1.0。
    - text_scale (float): 文本编码器模型的缩放因子，默认为1.0。
    - num_inference_steps (int): 生成图像的推理步数，默认为50。
    - guidance_scale (float): 指导尺度，控制生成图像的多样性，默认为5.0。
    - offset (int): 从第几个检查点开始，默认为0。
    - limit (int): 要处理的检查点数量，默认为10。
    - seed (int): 随机种子，用于结果可重复性，默认为0。

    返回:
    - imgs (List[PIL.Image.Image]): 生成的图像列表。
    """
    # 存储生成的图像
    imgs = []

    # 如果path_alls是字符串，则将其作为模式匹配的文件路径列表；否则，假设它已经是列表
    if isinstance(path_alls, str):
        # 获取匹配的文件路径列表，并去重
        alls = list(set(glob.glob(path_alls)))
        # 按修改时间排序
        alls.sort(key=os.path.getmtime)
    else:
        # 直接使用提供的列表
        alls = path_alls

    # 加载预训练的Stable Diffusion管道，使用float16精度以节省显存
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)

    # 输出找到的检查点数量
    print(f"Found {len(alls)} checkpoints")
    # 遍历指定范围内的检查点
    for path in alls[offset:limit]:
        # 输出当前检查点的路径
        print(path)

        # 对管道进行patch操作
        patch_pipe(
            pipe, path, patch_unet=patch_unet, patch_text=patch_text, patch_ti=patch_ti
        )

        # 微调LoRA缩放因子
        tune_lora_scale(pipe.unet, unet_scale)
        tune_lora_scale(pipe.text_encoder, text_sclae)

        # 设置随机种子以确保结果可重复
        torch.manual_seed(seed)

        # 生成图像
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        imgs.append(image)

    return imgs
