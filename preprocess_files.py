from typing import List, Literal, Union, Optional, Tuple
import os
from PIL import Image, ImageFilter
import torch
import numpy as np
import fire
from tqdm import tqdm
import glob
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


# 禁用梯度计算，节省显存并加快推理速度
@torch.no_grad()
def swin_ir_sr(
    images: List[Image.Image],
    model_id: Literal[
        "caidas/swin2SR-classical-sr-x2-64", "caidas/swin2SR-classical-sr-x4-48"
    ] = "caidas/swin2SR-classical-sr-x2-64",
    target_size: Optional[Tuple[int, int]] = None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[Image.Image]:
    """
    使用SwinIR模型对图像进行超分辨率处理，返回一个PIL图像列表。

    参数:
    - images (List[Image.Image]): 输入的PIL图像列表。
    - model_id (Literal["caidas/swin2SR-classical-sr-x2-64", "caidas/swin2SR-classical-sr-x4-48"]): 
        使用的SwinIR模型标识符。默认为 'caidas/swin2SR-classical-sr-x2-64'，表示2倍超分辨率。
        可选值:
            - "caidas/swin2SR-classical-sr-x2-64": 2倍超分辨率，图像尺寸为64x64。
            - "caidas/swin2SR-classical-sr-x4-48": 4倍超分辨率，图像尺寸为48x48。
    - target_size (Optional[Tuple[int, int]]): 目标图像尺寸。如果指定，只有尺寸小于目标尺寸的图像会被处理。
    - device (torch.device): 计算设备，默认为 'cuda:0' 如果有GPU可用，否则使用CPU。
    - **kwargs: 其他可选的关键字参数。

    返回:
    - out_images (List[Image.Image]): 处理后的PIL图像列表。
    """
    # 从transformers库中导入Swin2SR模型和处理器
    from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor

    # 加载预训练的SwinIR模型并移动到指定设备
    model = Swin2SRForImageSuperResolution.from_pretrained(
        model_id,
    ).to(device)
    # 初始化图像处理器
    processor = Swin2SRImageProcessor()

    # 存储输出图像
    out_images = []

    # 遍历输入图像
    for image in tqdm(images):
        # 获取原始图像尺寸
        ori_w, ori_h = image.size
        # 如果指定了目标尺寸，并且原始图像尺寸大于或等于目标尺寸，则跳过处理
        if target_size is not None:
            if ori_w >= target_size[0] and ori_h >= target_size[1]:
                out_images.append(image)
                continue
        
        # 使用处理器对图像进行预处理，并移动到指定设备
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            # 前向传播，获取模型输出
            outputs = model(**inputs)

        # 从输出中提取重建图像数据
        output = (
            outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        )
        # 将通道从第一维移动到最后一维
        output = np.moveaxis(output, source=0, destination=-1)
        # 将像素值缩放到0-255并转换为uint8类型
        output = (output * 255.0).round().astype(np.uint8)
        # 将numpy数组转换为PIL图像
        output = Image.fromarray(output)

        # 将处理后的图像添加到输出列表中
        out_images.append(output)

    return out_images


@torch.no_grad()
def clipseg_mask_generator(
    images: List[Image.Image],
    target_prompts: Union[List[str], str],
    model_id: Literal[
        "CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"
    ] = "CIDAS/clipseg-rd64-refined",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    bias: float = 0.01,
    temp: float = 1.0,
    **kwargs,
) -> List[Image.Image]:
    """
    为每个图像生成一个灰度掩码，其中掩码表示目标提示在图像中出现的概率。

    参数:
    - images (List[Image.Image]): 输入的PIL图像列表。
    - target_prompts (Union[List[str], str]): 目标提示，可以是字符串或字符串列表。如果为字符串，则对所有图像使用相同的提示。
    - model_id (Literal["CIDAS/clipseg-rd64-refined", "CIDAS/clipseg-rd16"]): 
        使用的CLIPSeg模型标识符。默认为 'CIDAS/clipseg-rd64-refined'。
        可选值:
            - "CIDAS/clipseg-rd64-refined": 64x64分辨率的精炼模型。
            - "CIDAS/clipseg-rd16": 16x16分辨率的模型。
    - device (torch.device): 计算设备，默认为 'cuda:0' 如果有GPU可用，否则使用CPU。
    - bias (float): 偏置值，添加到概率中以调整掩码的亮度。默认为0.01。
    - temp (float): 温度参数，用于调整softmax函数。默认为1.0。
    - **kwargs: 其他可选的关键字参数。

    返回:
    - masks (List[Image.Image]): 生成的灰度掩码列表。
    """
    # 如果只有一个提示，则将其重复应用到所有图像
    if isinstance(target_prompts, str):
        print(
            f'Warning: only one target prompt "{target_prompts}" was given, so it will be used for all images'
        )

        target_prompts = [target_prompts] * len(images)

    # 从transformers库中导入CLIPSeg模型和处理器
    processor = CLIPSegProcessor.from_pretrained(model_id)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(device)

    # 存储输出掩码
    masks = []
    # 遍历图像和提示的组合
    for image, prompt in tqdm(zip(images, target_prompts)):
        # 获取原始图像尺寸
        original_size = image.size

        # 使用处理器对图像和文本进行预处理，并移动到指定设备
        inputs = processor(
            text=[prompt, ""],
            images=[image] * 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        # 前向传播，获取模型输出
        outputs = model(**inputs)

        # 从输出中提取logits
        logits = outputs.logits
        # 应用softmax函数并调整温度
        probs = torch.nn.functional.softmax(logits / temp, dim=0)[0]
        # 添加偏置并限制在0到1之间
        probs = (probs + bias).clamp_(0, 1)
        # 将概率缩放到0-255
        probs = 255 * probs / probs.max()

        # 将概率转换为灰度图像
        mask = Image.fromarray(probs.cpu().numpy()).convert("L")

        # 将掩码调整回原始图像尺寸
        mask = mask.resize(original_size)
        
        # 将生成的掩码添加到列表中
        masks.append(mask)

    return masks


@torch.no_grad()
def blip_captioning_dataset(
    images: List[Image.Image],
    text: Optional[str] = None,
    model_id: Literal[
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-image-captioning-base",
    ] = "Salesforce/blip-image-captioning-large",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    **kwargs,
) -> List[str]:
    """
    为给定的图像列表生成字幕列表。

    参数:
    - images (List[Image.Image]): 输入的PIL图像列表。
    - text (Optional[str]): 可选的文本提示，用于引导字幕生成。如果提供，模型将基于该提示生成字幕。
    - model_id (Literal["Salesforce/blip-image-captioning-large", "Salesforce/blip-image-captioning-base"]): 
        使用的BLIP模型标识符。默认为 'Salesforce/blip-image-captioning-large'，表示使用大型模型。
        可选值:
            - "Salesforce/blip-image-captioning-large": 大型模型，适用于需要高质量字幕的任务。
            - "Salesforce/blip-image-captioning-base": 基础模型，适用于资源有限或对速度要求较高的任务。
    - device (torch.device): 计算设备，默认为 'cuda' 如果有GPU可用，否则使用CPU。
    - **kwargs: 其他可选的关键字参数。

    返回:
    - captions (List[str]): 生成的字幕列表，每个元素对应输入图像列表中的一个图像。
    """
    # 从transformers库中导入BLIP模型和处理器
    from transformers import BlipProcessor, BlipForConditionalGeneration

    # 加载预训练的BLIP模型和处理器
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    # 存储生成的字幕
    captions = []

    # 遍历输入图像
    for image in tqdm(images):
        # 使用处理器对图像和文本进行预处理，并移动到指定设备
        inputs = processor(image, text=text, return_tensors="pt").to("cuda")
        # 使用模型生成字幕，设置最大长度、采样参数等
        out = model.generate(
            **inputs, max_length=150, do_sample=True, top_k=50, temperature=0.7
        )
        # 解码生成的字幕，跳过特殊标记
        caption = processor.decode(out[0], skip_special_tokens=True)
        # 将生成的字幕添加到列表中
        captions.append(caption)
    # 返回字幕列表
    return captions


def face_mask_google_mediapipe(
    images: List[Image.Image], blur_amount: float = 80.0, bias: float = 0.05
) -> List[Image.Image]:
    """
    返回带有面部掩码的图像列表。

    参数:
    - images (List[Image.Image]): 输入的PIL图像列表。
    - blur_amount (float): 模糊量，用于模糊面部区域。默认为80.0。
    - bias (float): 偏置值，用于调整掩码的亮度。默认为0.05。

    返回:
    - masks (List[Image.Image]): 带有面部掩码的图像列表。
    """
    import mediapipe as mp

    # 导入面部分析模块
    mp_face_detection = mp.solutions.face_detection

    # 初始化面部分析器，设置模型选择和最小检测置信度
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    # 存储生成的掩码图像
    masks = []
    # 遍历输入图像
    for image in tqdm(images):
        # 将PIL图像转换为numpy数组
        image = np.array(image)

        # 进行面部分析
        results = face_detection.process(image)
        # 创建一个黑色图像作为掩码
        black_image = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        if results.detections:
            # 如果检测到面部，则绘制白色矩形覆盖面部区域
            for detection in results.detections:
                # 获取面部边界框的相对坐标
                # 将相对坐标转换为像素坐标
                x_min = int(
                    detection.location_data.relative_bounding_box.xmin * image.shape[1]
                )
                y_min = int(
                    detection.location_data.relative_bounding_box.ymin * image.shape[0]
                )
                width = int(
                    detection.location_data.relative_bounding_box.width * image.shape[1]
                )
                height = int(
                    detection.location_data.relative_bounding_box.height
                    * image.shape[0]
                )

                # 在掩码图像上绘制白色矩形
                black_image[y_min : y_min + height, x_min : x_min + width] = 255

        # 将numpy数组转换回PIL图像
        black_image = Image.fromarray(black_image)
        # 将掩码图像添加到列表中
        masks.append(black_image)

    # 返回掩码图像列表
    return masks


def _crop_to_square(
    image: Image.Image, com: List[Tuple[int, int]], resize_to: Optional[int] = None
):
    """
    将图像裁剪为正方形，基于给定的中心点。

    参数:
    - image (Image.Image): 输入的PIL图像。
    - com (List[Tuple[int, int]]): 中心点坐标列表，每个元素是一个元组 (cx, cy)。
    - resize_to (Optional[int]): 可选的裁剪后调整大小参数。如果提供，图像将被调整为该大小。

    返回:
    - cropped_image (Image.Image): 裁剪后的正方形图像。
    """
    # 获取中心点坐标
    cx, cy = com
    # 获取图像的宽度和高度
    width, height = image.size

    if width > height:
        # 如果宽度大于高度，则在宽度方向上裁剪
        # 计算左侧可能的起始位置
        left_possible = max(cx - height / 2, 0)
        # 确保不超过图像边界
        left = min(left_possible, width - height)
        # 计算右侧位置
        right = left + height
        # 上侧位置
        top = 0
        # 下侧位置
        bottom = height
    else:
        # 如果高度大于或等于宽度，则在高度方向上裁剪
        # 左侧位置
        left = 0
        # 右侧位置
        right = width
        # 计算顶部可能的起始位置
        top_possible = max(cy - width / 2, 0)
        # 确保不超过图像边界
        top = min(top_possible, height - width)
        # 计算底部位置
        bottom = top + width

    # 裁剪图像
    image = image.crop((left, top, right, bottom))

    if resize_to:
        # 如果提供了调整大小参数，则将图像调整为指定大小
        image = image.resize((resize_to, resize_to), Image.Resampling.LANCZOS)

    # 返回裁剪后的图像
    return image


def _center_of_mass(mask: Image.Image):
    """
    计算图像掩码的中心点坐标。

    参数:
    - mask (Image.Image): 输入的PIL图像掩码，掩码中非零像素表示感兴趣区域。

    返回:
    - Tuple[float, float]: 中心点坐标 (x, y)。
    """
    # 将掩码转换为numpy数组
    x, y = np.meshgrid(np.arange(mask.size[0]), np.arange(mask.size[1]))
    
    # 创建x和y坐标网格
    x_ = x * np.array(mask)
    y_ = y * np.array(mask)

    x = np.sum(x_) / np.sum(mask)
    y = np.sum(y_) / np.sum(mask)

    return x, y


def load_and_save_masks_and_captions(
    files: Union[str, List[str]],
    output_dir: str,
    caption_text: Optional[str] = None,
    target_prompts: Optional[Union[List[str], str]] = None,
    target_size: int = 512,
    crop_based_on_salience: bool = True,
    use_face_detection_instead: bool = False,
    temp: float = 1.0,
    n_length: int = -1,
):
    """
    从给定的文件路径加载图像，为每张图像生成掩码，并保存掩码、字幕和超分辨率图像到输出目录。

    参数:
    - files (Union[str, List[str]]): 输入文件路径，可以是单个字符串（目录路径或文件路径）或文件路径列表。
    - output_dir (str): 输出目录路径，用于保存处理后的图像和掩码。
    - caption_text (Optional[str]): 可选的文本提示，用于引导字幕生成。如果提供，模型将基于该提示生成字幕。
    - target_prompts (Optional[Union[List[str], str]]): 目标提示，可以是字符串或字符串列表。如果未提供，则使用生成的字幕作为目标提示。
    - target_size (int): 目标图像大小，默认为512。图像将被调整为该大小。
    - crop_based_on_salience (bool): 是否基于显著性裁剪图像。如果为True，则根据掩码的中心点进行裁剪；否则，裁剪到图像中心。
    - use_face_detection_instead (bool): 是否使用面部分割代替语义分割。如果为True，则使用MediaPipe进行面部分割；否则，使用CLIPSeg进行语义分割。
    - temp (float): 温度参数，用于调整CLIPSeg模型的softmax函数。默认为1.0。
    - n_length (int): 要处理的图像数量。如果为-1，则处理所有图像。
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 加载图像
    if isinstance(files, str):
        # 如果是字符串，检查是否为目录
        if os.path.isdir(files):
            # 获取目录中的所有.png和.jpg文件
            files = glob.glob(os.path.join(files, "*.png")) + glob.glob(
                os.path.join(files, "*.jpg")
            )

        if len(files) == 0:
            # 如果没有找到文件，则抛出异常
            raise Exception(
                f"No files found in {files}. Either {files} is not a directory or it does not contain any .png or .jpg files."
            )
        if n_length == -1:
            # 如果未指定长度，则处理所有文件
            n_length = len(files)
        # 按排序顺序选择前n_length个文件
        files = sorted(files)[:n_length]

    images = [Image.open(file) for file in files]

    # 生成captions
    print(f"Generating {len(images)} captions...")
    # 使用BLIP模型生成字幕
    captions = blip_captioning_dataset(images, text=caption_text)

    if target_prompts is None:
        # 如果未提供目标提示，则使用字幕作为目标提示
        target_prompts = captions

    print(f"Generating {len(images)} masks...")
    if not use_face_detection_instead:
        # 使用CLIPSeg模型生成语义分割掩码
        seg_masks = clipseg_mask_generator(
            images=images, target_prompts=target_prompts, temp=temp
        )
    else:
        # 使用MediaPipe进行面部分割
        seg_masks = face_mask_google_mediapipe(images=images)

    # 计算掩码的中心点
    if crop_based_on_salience:
        # 基于掩码计算中心点
        coms = [_center_of_mass(mask) for mask in seg_masks]
    else:
        # 使用图像中心作为中心点
        coms = [(image.size[0] / 2, image.size[1] / 2) for image in images]
    # 根据中心点裁剪图像到正方形
    images = [
        _crop_to_square(image, com, resize_to=None) for image, com in zip(images, coms)
    ]

    print(f"Upscaling {len(images)} images...")
    # 对图像进行超分辨率处理
    # 使用SwinIR模型进行超分辨率处理
    images = swin_ir_sr(images, target_size=(target_size, target_size))
    images = [
        image.resize((target_size, target_size), Image.Resampling.LANCZOS) # 使用高质量的重采样方法调整图像大小
        for image in images
    ]

    # 根据中心点裁剪掩码到目标大小
    seg_masks = [
        _crop_to_square(mask, com, resize_to=target_size)
        for mask, com in zip(seg_masks, coms)
    ]
    # 保存字幕到文件
    with open(os.path.join(output_dir, "caption.txt"), "w") as f:
        # 保存图像和掩码到输出目录
        for idx, (image, mask, caption) in enumerate(zip(images, seg_masks, captions)):
            # 保存超分辨率图像
            image.save(os.path.join(output_dir, f"{idx}.src.jpg"), quality=99)
            # 保存掩码图像
            mask.save(os.path.join(output_dir, f"{idx}.mask.png"))
            # 将字幕写入文件
            f.write(caption + "\n")


def main():
    # 使用Fire库将函数转换为命令行接口
    fire.Fire(load_and_save_masks_and_captions)
