import json
from pathlib import Path


def calculate_average_usage(directory_path):
    """
    计算指定目录下所有JSON文件的平均usage信息
    
    Args:
        directory_path: JSON文件所在的目录路径
        
    Returns:
        dict: 包含平均usage信息的字典
    """
    # 初始化统计变量
    total_completion_tokens = 0
    total_prompt_tokens = 0
    total_tokens = 0
    total_n_completions = 0
    file_count = 0

    # 存储每个文件的usage信息(可选,用于详细分析)
    file_usages = []

    # 遍历目录下所有JSON文件
    json_files = list(Path(directory_path).glob('*.json'))

    print(f"找到 {len(json_files)} 个JSON文件")
    print("=" * 60)

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取usage信息
            if 'usage' in data:
                usage = data['usage']

                completion_tokens = usage.get('completion_tokens', 0)
                prompt_tokens = usage.get('prompt_tokens', 0)
                tokens = usage.get('total_tokens', 0)
                n_completions = usage.get('n_completions', 0)

                # 累加
                total_completion_tokens += completion_tokens
                total_prompt_tokens += prompt_tokens
                total_tokens += tokens
                total_n_completions += n_completions
                file_count += 1

                # 保存单个文件的usage信息
                file_usages.append({
                    'filename': json_file.name,
                    'completion_tokens': completion_tokens,
                    'prompt_tokens': prompt_tokens,
                    'total_tokens': tokens,
                    'n_completions': n_completions
                })

        except (json.JSONDecodeError, KeyError) as e:
            print(f"警告: 处理文件 {json_file.name} 时出错: {e}")
            continue

    # 计算平均值
    if file_count > 0:
        avg_usage = {
            'file_count': file_count,
            'avg_completion_tokens': total_completion_tokens / file_count,
            'avg_prompt_tokens': total_prompt_tokens / file_count,
            'avg_total_tokens': total_tokens / file_count,
            'avg_n_completions': total_n_completions / file_count,
            'total_completion_tokens': total_completion_tokens,
            'total_prompt_tokens': total_prompt_tokens,
            'total_tokens_all_files': total_tokens,
            'total_n_completions': total_n_completions
        }

        # 打印结果
        print("\n平均Usage统计:")
        print("=" * 60)
        print(f"处理的文件数量: {avg_usage['file_count']}")
        print("-" * 60)
        print(f"{avg_usage['avg_n_completions']:.2f}", end=" | ")
        print(f"{avg_usage['avg_prompt_tokens']:.2f}", end=" | ")
        print(f"{avg_usage['avg_completion_tokens']:.2f}", end=" | ")

        return avg_usage, file_usages
    else:
        print("未找到有效的usage数据")
        return None, []


if __name__ == "__main__":
    # 设置目录路径
    directory = "cache/mimic3-drugrec/MedIKAL/0shot-deepseek-base-kg"
    avg_usage, file_usages = calculate_average_usage(directory)

    directory = "cache/mimic4-drugrec/MedIKAL/0shot-deepseek-base-kg"
    avg_usage, file_usages = calculate_average_usage(directory)

    directory = "cache/mimic3-diagnosis/MedIKAL/0shot-deepseek-base-kg"
    avg_usage, file_usages = calculate_average_usage(directory)

    directory = "cache/mimic4-diagnosis/MedIKAL/0shot-deepseek-base-kg"
    avg_usage, file_usages = calculate_average_usage(directory)
