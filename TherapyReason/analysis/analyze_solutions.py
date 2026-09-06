import json


def analyze_solutions(file_path):
    """
    分析solution_thres4.jsonl文件中的solution数量以及每个solution的conditions、procedures和drugs的平均数
    """
    
    total_solutions = 0
    total_conditions = 0
    total_procedures = 0
    total_drugs = 0
    total_candidates = 0

    line_count = 0
    solutions_per_line = []
    conditions_list = []
    procedures_list = []
    drugs_list = []
    candidates_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            line_count += 1
            data = json.loads(line)
            
            # 获取solutions列表
            solutions = data.get('solutions', [])

            solutions = [x for y in solutions for x in y]


            
            # 统计当前行的solution数量
            num_solutions = len(solutions)
            total_solutions += num_solutions
            solutions_per_line.append(num_solutions)
            
            # 遍历每个solution,统计conditions、procedures和drugs的数量
            for solution in solutions:
                conditions = solution.get('conditions', [])
                procedures = solution.get('procedures', [])
                drugs = solution.get('drugs', [])
                candidates = solution.get('candidates', [])

                num_conditions = len(conditions)
                num_procedures = len(procedures)
                num_drugs = len(drugs)
                num_candidates = len(candidates)

                total_conditions += num_conditions
                total_procedures += num_procedures
                total_drugs += num_drugs
                total_candidates += num_candidates

                conditions_list.append(num_conditions)
                procedures_list.append(num_procedures)
                drugs_list.append(num_drugs)
                # Record this solution's candidate count, not the cumulative
                # total across all previously processed solutions.
                candidates_list.append(num_candidates)

    # 计算平均数
    avg_solutions_per_line = total_solutions / line_count if line_count > 0 else 0
    avg_conditions = total_conditions / total_solutions if total_solutions > 0 else 0
    avg_procedures = total_procedures / total_solutions if total_solutions > 0 else 0
    avg_drugs = total_drugs / total_solutions if total_solutions > 0 else 0
    avg_candidates = total_candidates / total_solutions if total_solutions > 0 else 0

    print("=" * 70)
    print("Solution 统计分析结果")
    print("=" * 70)
    print(f"\n文件路径: {file_path}")
    print(f"总行数: {line_count}")
    print(f"总solution数量: {total_solutions}")
    print(f"\n平均每行solution数量: {avg_solutions_per_line:.2f}")
    print(f"  - 最小值: {min(solutions_per_line) if solutions_per_line else 0}")
    print(f"  - 最大值: {max(solutions_per_line) if solutions_per_line else 0}")
    
    print(f"\n{'=' * 70}")
    print("每个solution的平均组成:")
    print(f"{'=' * 70}")
    print(f"  conditions (条件):")
    print(f"    - 平均数: {avg_conditions:.2f}")
    print(f"    - 最小值: {min(conditions_list) if conditions_list else 0}")
    print(f"    - 最大值: {max(conditions_list) if conditions_list else 0}")
    
    print(f"\n  procedures (程序):")
    print(f"    - 平均数: {avg_procedures:.2f}")
    print(f"    - 最小值: {min(procedures_list) if procedures_list else 0}")
    print(f"    - 最大值: {max(procedures_list) if procedures_list else 0}")
    
    print(f"\n  drugs (药物):")
    print(f"    - 平均数: {avg_drugs:.2f}")
    print(f"    - 最小值: {min(drugs_list) if drugs_list else 0}")
    print(f"    - 最大值: {max(drugs_list) if drugs_list else 0}")

    print(f"\n  candidates (候选):")
    print(f"    - 平均数: {avg_candidates:.2f}")
    print(f"    - 最小值: {min(candidates_list) if candidates_list else 0}")
    print(f"    - 最大值: {max(candidates_list) if candidates_list else 0}")

    print(f"\n{'=' * 70}")
    print(f"每个solution的总项目数 (conditions + procedures + drugs):")
    print(f"{'=' * 70}")
    total_items = [c + p + d for c, p, d in zip(conditions_list, procedures_list, drugs_list)]
    avg_total = sum(total_items) / len(total_items) if total_items else 0
    print(f"  - 平均数: {avg_total:.2f}")
    print(f"  - 最小值: {min(total_items) if total_items else 0}")
    print(f"  - 最大值: {max(total_items) if total_items else 0}")
    print("=" * 70)
    
    return {
        'total_lines': line_count,
        'total_solutions': total_solutions,
        'avg_solutions_per_line': avg_solutions_per_line,
        'avg_conditions': avg_conditions,
        'avg_procedures': avg_procedures,
        'avg_drugs': avg_drugs,
        'avg_candidates': avg_candidates,
        'min_solutions_per_line': min(solutions_per_line) if solutions_per_line else 0,
        'max_solutions_per_line': max(solutions_per_line) if solutions_per_line else 0,
        'min_candidates': min(candidates_list) if candidates_list else 0,
        'max_candidates': max(candidates_list) if candidates_list else 0,
    }

if __name__ == '__main__':
    file_path = './TherapyReason/cache/mimic3-diagnosis/sensitivity/solution_thres4.jsonl'
    result = analyze_solutions(file_path)
