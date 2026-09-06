import json
import os
from pathlib import Path

def extract_usage_from_json(file_path):
    """从单个JSON文件中提取usage信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_completion_tokens = 0
        total_prompt_tokens = 0
        total_tokens = 0
        total_n_completions = 0
        
        # 遍历数据结构
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, dict):
                            # 查找所有包含usage的字段
                            for key in sub_item:
                                if 'usage' in key.lower() and isinstance(sub_item[key], dict):
                                    usage = sub_item[key]
                                    total_completion_tokens += usage.get('completion_tokens', 0)
                                    total_prompt_tokens += usage.get('prompt_tokens', 0)
                                    total_tokens += usage.get('total_tokens', 0)
                                    total_n_completions += usage.get('n_completions', 0)
        
        return {
            'completion_tokens': total_completion_tokens,
            'prompt_tokens': total_prompt_tokens,
            'total_tokens': total_tokens,
            'n_completions': total_n_completions
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_directory(dir_path):
    """处理目录下的所有JSON文件"""
    dir_path = Path(dir_path)
    usage_dict = {}
    
    if not dir_path.exists():
        print(f"Directory not found: {dir_path}")
        return usage_dict
    
    # 遍历目录下所有json文件
    for json_file in sorted(dir_path.glob('*.json')):
        sample_id = json_file.stem  # 文件名不带扩展名
        usage = extract_usage_from_json(json_file)
        
        if usage is not None:
            usage_dict[sample_id] = usage
    
    return usage_dict

def main():
    base_path = "./TherapyReason/cache/mimic4-drugrec/sensitivity"
    
    # 定义三个目录
    directories = {
        'dl': os.path.join(base_path, 'solution_thres4_dl/deepseek'),
        'kg': os.path.join(base_path, 'solution_thres4_kg_2shot/deepseek'),
        'overview': os.path.join(base_path, 'solution_thres4_overview/deepseek')
    }
    
    # 存储每个目录的usage数据
    all_usages = {}
    
    # 处理每个目录
    for name, dir_path in directories.items():
        print(f"\nProcessing {name} directory...")
        usage_data = process_directory(dir_path)
        all_usages[name] = usage_data
        print(f"  Found {len(usage_data)} samples")
        
        # 计算该目录的平均usage
        if usage_data:
            avg_completion = sum(u['completion_tokens'] for u in usage_data.values()) / len(usage_data)
            avg_prompt = sum(u['prompt_tokens'] for u in usage_data.values()) / len(usage_data)
            avg_total = sum(u['total_tokens'] for u in usage_data.values()) / len(usage_data)
            avg_n_completions = sum(u['n_completions'] for u in usage_data.values()) / len(usage_data)
            
            print(f"  Average completion tokens: {avg_completion:.2f}")
            print(f"  Average prompt tokens: {avg_prompt:.2f}")
            print(f"  Average total tokens: {avg_total:.2f}")
            print(f"  Average n_completions: {avg_n_completions:.2f}")
    
    # 找到所有目录中都存在的sample
    common_samples = set(all_usages['dl'].keys()) & set(all_usages['kg'].keys()) & set(all_usages['overview'].keys())
    print(f"\n{'='*60}")
    print(f"Common samples across all directories: {len(common_samples)}")
    print(f"{'='*60}\n")
    
    # 对相同的sample求和usage
    summed_usage = {}
    for sample_id in common_samples:
        summed_usage[sample_id] = {
            'completion_tokens': sum(all_usages[dir_name][sample_id]['completion_tokens'] 
                                    for dir_name in ['dl', 'kg', 'overview']),
            'prompt_tokens': sum(all_usages[dir_name][sample_id]['prompt_tokens'] 
                                for dir_name in ['dl', 'kg', 'overview']),
            'total_tokens': sum(all_usages[dir_name][sample_id]['total_tokens'] 
                               for dir_name in ['dl', 'kg', 'overview']),
            'n_completions': sum(all_usages[dir_name][sample_id]['n_completions'] 
                                for dir_name in ['dl', 'kg', 'overview'])
        }
    
    # 计算所有sample的平均usage（求和后的）
    if summed_usage:
        overall_avg_completion = sum(u['completion_tokens'] for u in summed_usage.values()) / len(summed_usage)
        overall_avg_prompt = sum(u['prompt_tokens'] for u in summed_usage.values()) / len(summed_usage)
        overall_avg_total = sum(u['total_tokens'] for u in summed_usage.values()) / len(summed_usage)
        overall_avg_n_completions = sum(u['n_completions'] for u in summed_usage.values()) / len(summed_usage)
        
        print("Overall Average Usage (summed across all three directories for each sample):")
        print(f"  Average completion tokens: {overall_avg_completion:.2f}")
        print(f"  Average prompt tokens: {overall_avg_prompt:.2f}")
        print(f"  Average total tokens: {overall_avg_total:.2f}")
        print(f"  Average n_completions: {overall_avg_n_completions:.2f}")
        print(f"\nTotal samples processed: {len(summed_usage)}")
        
        # 可选：保存详细结果到文件
        output_file = "usage_statistics.json"
        result = {
            'directory_stats': {},
            'overall_average': {
                'completion_tokens': overall_avg_completion,
                'prompt_tokens': overall_avg_prompt,
                'total_tokens': overall_avg_total,
                'n_completions': overall_avg_n_completions,
                'num_samples': len(summed_usage)
            },
            'per_sample_summed_usage': summed_usage
        }
        
        for name in ['dl', 'kg', 'overview']:
            usage_data = all_usages[name]
            if usage_data:
                avg_completion = sum(u['completion_tokens'] for u in usage_data.values()) / len(usage_data)
                avg_prompt = sum(u['prompt_tokens'] for u in usage_data.values()) / len(usage_data)
                avg_total = sum(u['total_tokens'] for u in usage_data.values()) / len(usage_data)
                avg_n_completions = sum(u['n_completions'] for u in usage_data.values()) / len(usage_data)
                result['directory_stats'][name] = {
                    'num_samples': len(usage_data),
                    'avg_completion_tokens': avg_completion,
                    'avg_prompt_tokens': avg_prompt,
                    'avg_total_tokens': avg_total,
                    'avg_n_completions': avg_n_completions
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
    else:
        print("No common samples found across all directories!")

if __name__ == "__main__":
    main()
