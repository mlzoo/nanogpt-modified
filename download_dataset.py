"""
下载 HuggingFace 数据集 MLZoo/edu-fineweb-10B
这个数据集包含 .npy 格式的 token 文件
"""
import os
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

def download_edu_fineweb(download_all=False):
    """
    下载 MLZoo/edu-fineweb-10B 数据集
    这是一个教育质量的 FineWeb 数据集，包含10B tokens，存储为 .npy 文件
    
    参数:
        download_all: 是否下载全部文件（默认 False）
    """
    print("=" * 60)
    print("开始下载 MLZoo/edu-fineweb-10B 数据集")
    print("=" * 60)
    print("\n数据集信息:")
    print("- 这个数据集包含约 10B tokens")
    print("- 数据存储为 .npy 格式（NumPy 数组）")
    print("- 每个文件包含预分词的 token IDs")
    print()
    
    # 设置缓存目录
    cache_dir = "./edu_fineweb10B"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 获取所有 .npy 文件列表
        print("正在获取数据集文件列表...")
        all_files = list_repo_files("MLZoo/edu-fineweb-10B", repo_type="dataset")
        npy_files = sorted([f for f in all_files if f.endswith('.npy')])
        
        print(f"✓ 找到 {len(npy_files)} 个 .npy 文件")
        
        # 确定要下载的文件
        files_to_download = npy_files
        
        print("\n开始下载文件...")
        downloaded_files = []
        
        # 下载文件
        for filename in tqdm(files_to_download, desc="下载进度"):
            try:
                local_path = hf_hub_download(
                    repo_id="MLZoo/edu-fineweb-10B",
                    filename=filename,
                    repo_type="dataset",
                    local_dir=cache_dir
                )
                downloaded_files.append(local_path)
            except Exception as e:
                print(f"\n下载 {filename} 时出错: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("✓ 下载完成！")
        print("=" * 60)
        print(f"\n已下载 {len(downloaded_files)} 个文件到: {cache_dir}")
        
        if downloaded_files:
            first_file = downloaded_files[0]
            try:
                data = np.load(first_file)
                total_tokens = 0
                for f in downloaded_files:
                    try:
                        data = np.load(f)
                        total_tokens += len(data)
                    except:
                        pass
                
                print(f"\n已下载的总 token 数: {total_tokens:,}")
                
            except Exception as e:
                print(f"  读取文件时出错: {e}")
        
        print("```")
        
        return downloaded_files
        
    except Exception as e:
        print(f"\n下载过程中出现错误: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. HuggingFace 访问受限")
        print("3. 磁盘空间不足")
        print("\n建议:")
        print("- 检查网络连接")
        print("- 确保有足够的磁盘空间")
        return None

if __name__ == "__main__":
    # 默认下载前 5 个文件
    # 如需下载全部，请使用: download_edu_fineweb(download_all=True)
    downloaded_files = download_edu_fineweb(download_all=False)
    
    if downloaded_files:
        print("\n✓ 数据集准备完成！可以开始训练了。")
