"""
实验数据集下载脚本

修改下方的配置以下载你的实验数据集
"""

from aistudio_dataset_downloader import AISTUDIO_DatasetDownloader

# ==================== 配置区域 ====================

# 数据集 ID (从 AI Studio 数据集页面获取)
DATASET_ID = "213196/ssWnRtyD"  # 替换为你的数据集ID

# 本地保存路径
LOCAL_DIR = "./data/experiment_dataset"

# Access Token (仅私密数据集需要)
# 在我的控制台 -> 令牌管理 中获取
ACCESS_TOKEN = None  # 如果需要，填入你的 token

# ==================== 执行下载 ====================

if __name__ == "__main__":
    # 创建下载器
    downloader = AISTUDIO_DatasetDownloader(
        dataset_id=DATASET_ID,
        local_dir=LOCAL_DIR,
        token=ACCESS_TOKEN
    )
    
    # 下载完整数据集
    success = downloader.download_full_dataset()
    
    if success:
        print("\n✅ 数据集准备完成！")
        print(f"📍 数据位置: {LOCAL_DIR}")
        print("\n下一步: 创建数据集加载器并集成到训练流程")
    else:
        print("\n❌ 下载失败，请检查配置和网络连接")
