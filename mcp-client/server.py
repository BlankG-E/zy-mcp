
import os
from dotenv import load_dotenv
import json
import cv2
from typing import Any,Dict
from mcp.server.fastmcp import FastMCP
import numpy as np
import re
import base64

mcp = FastMCP("VdieoDescribe")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")
model = os.getenv("MODEL")

async def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def extract_video_path(query: str) -> str:
    pattern = r'([a-zA-Z]:\\[^<>:"|?*\n\r]*\.(mp4|avi|mov|mkv|flv))'
    match = re.search(pattern, query)
    if match:
        return match.group(1)
    else:
        return None
    

    

async def video_inference_model(video_path: str) -> Dict[str, Any]:
    """
    对视频进行推理，返回推理结果。
    :param video_path: 视频文件路径
    :return: 推理结果字典
    """
    # 这里使用 OpenCV 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "无法打开视频文件"}
    
    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # 这里只是一个示例，实际推理过程会更复杂
    # 假设我们对视频的每一帧进行简单的颜色分析
    color_analysis = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 计算每一帧的平均颜色
            avg_color = np.mean(frame, axis=(0, 1))
            color_analysis.append(avg_color.tolist())
    finally:
        cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "color_analysis": color_analysis
    }

async def video_binarization(video_path: str, threshold: int = 127) -> Dict[str, Any]:
    """
    对图像进行二值化处理，返回二值化后的图像路径。
    :param video_path: 图像文件路径
    :param threshold: sb二值化阈值
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "无法打开视频文件"}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps

    binary_stats = {
        "white_pixel_counts":[],
        "black_pixel_counts":[],
        "frame_brightness":[]
    }
    processed_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为灰度图像
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 进行二值化处理
            _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
            
            # 统计二值化结果
            white_pixels = np.sum(binary_frame == 255)
            black_pixels = np.sum(binary_frame == 0)
            avg_brightness = np.mean(gray_frame)
            
            binary_stats["white_pixel_counts"].append(int(white_pixels))
            binary_stats["black_pixel_counts"].append(int(black_pixels))
            binary_stats["frame_brightness"].append(float(avg_brightness))
            
            processed_frames += 1
            
    finally:
        cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "resolution": f"{width}x{height}",
        "binary_statistics": binary_stats,
        "success": True,
        "note": "二值化处理完成，未保存到本地文件"
    }
    




@mcp.tool()
async def inference_video(path: str) -> Dict[str, Any]:
    """
    对视频进行推理，返回推理结果。
    :param video_path: 视频文件路径
    :return: 推理结果字典
    """
    #video_path = await extract_video_path(path)
    #base64_video = encode_image(video_path)
    result = await video_inference_model(path)
    return result
@mcp.tool()
async def process_video_binarization(path: str, threshold: int = 127) -> Dict[str, Any]:
    """
    对视频进行二值化处理并返回统计信息
    :param path: 视频文件路径
    :param threshold: sb二值化阈值，范围0-255，默认为127
    :return: 视频二值化处理统计结果
    """
    result = await video_binarization(path, threshold)
    return result

if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')