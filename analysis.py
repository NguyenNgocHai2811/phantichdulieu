import pyspark
from pyspark.sql import SparkSession
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator

def parse_line(line):
    """
    Parses a single line of the CSV data.
    """
    try:
        parts = line.split(',')
        if len(parts) < 2: # Basic validation
            raise IndexError("Line is too short to be valid")

        sbd = parts[-1]
        scores_str = parts[1:-1]
        scores = []
        for s in scores_str:
            if s == "" or s.isalpha(): # Also handle cases like 'N1'
                scores.append(None)
            else:
                try:
                    scores.append(float(s))
                except ValueError:
                    scores.append(None)
        return (sbd, scores)
    except IndexError: # Catch invalid lines that don't split correctly
        return (None, [])

def main():
    """
    Main function to run the entire analysis.
    """
    spark = SparkSession.builder.appName("ExamAnalysis").getOrCreate()
    sc = spark.sparkContext
    data_file = "data/diemthi2020.csv"
    lines_rdd = sc.textFile(data_file)
    header = lines_rdd.first()
    subject_names = header.split(',')[1:-1]
    data_rdd = lines_rdd.filter(lambda row: row != header)

    parsed_rdd = data_rdd.map(parse_line).filter(lambda x: x[0] is not None and len(x[1]) > 0).cache()

    print("--- Bắt đầu phân tích dữ liệu toàn diện ---")

    # Câu 1: Đếm số bài thi điểm 0 từng môn.
    subject_map_by_index = {i: name for i, name in enumerate(subject_names)}
    zero_scores = parsed_rdd.flatMap(lambda p: [(i, p[1][i]) for i in range(len(p[1]))]) \
                            .filter(lambda x: x[1] is not None and x[1] == 0.0) \
                            .countByKey()
    print("\nCâu 1: Số bài thi có điểm 0 của từng môn:")
    for subject_index, count in sorted(zero_scores.items()):
        print(f"- {subject_map_by_index.get(subject_index, 'Unknown')}: {count} bài thi")

    # Câu 2: Môn thi có nhiều bài thi đạt điểm 10 nhất.
    ten_scores = parsed_rdd.flatMap(lambda p: [(i, p[1][i]) for i in range(len(p[1]))]) \
                           .filter(lambda x: x[1] is not None and x[1] == 10.0) \
                           .countByKey()
    if ten_scores:
        most_tens_subject_index = max(ten_scores, key=ten_scores.get)
        subject_name = subject_map_by_index.get(most_tens_subject_index, 'Unknown')
        count = ten_scores[most_tens_subject_index]
        print(f"\nCâu 2: Môn có nhiều điểm 10 nhất là '{subject_name}' với {count} bài thi.")
    else:
        print("\nCâu 2: Không có bài thi nào đạt điểm 10.")

    # Câu 3: SBD của thí sinh có điểm trung bình cao nhất.
    avg_scores_rdd = parsed_rdd.map(lambda p: (p[0], [s for s in p[1] if s is not None])) \
                               .map(lambda p: (p[0], sum(p[1]) / len(p[1]) if len(p[1]) > 0 else 0))
    highest_avg_score = avg_scores_rdd.map(lambda p: p[1]).max()
    top_students = avg_scores_rdd.filter(lambda p: p[1] == highest_avg_score).map(lambda p: p[0]).collect()
    print(f"\nCâu 3: Điểm trung bình cao nhất là: {highest_avg_score:.2f}")
    print(f"Số báo danh của các thí sinh đạt điểm trung bình cao nhất: {', '.join(top_students)}")

    # Câu 4: Thống kê số lượng thí sinh theo số môn thi.
    subjects_count_rdd = parsed_rdd.map(lambda p: (len([s for s in p[1] if s is not None]), 1)) \
                                   .reduceByKey(operator.add).sortByKey()
    print("\nCâu 4: Thống kê số lượng thí sinh theo số môn thi:")
    for num_subjects, count in subjects_count_rdd.collect():
        print(f"- {num_subjects} môn: {count} thí sinh")

    # Câu 5: Tính điểm trung bình bài thi Tự nhiên, Xã hội.
    natural_sciences_indices = [subject_names.index(s) for s in ['Li', 'Hoa', 'Sinh']]
    social_sciences_indices = [subject_names.index(s) for s in ['Su', 'Dia', 'GDCD']]
    natural_scores_rdd = parsed_rdd.map(lambda p: [p[1][i] for i in natural_sciences_indices]) \
                                   .filter(lambda scores: all(s is not None for s in scores)).map(lambda scores: sum(scores))
    social_scores_rdd = parsed_rdd.map(lambda p: [p[1][i] for i in social_sciences_indices]) \
                                  .filter(lambda scores: all(s is not None for s in scores)).map(lambda scores: sum(scores))
    avg_natural = natural_scores_rdd.mean() if not natural_scores_rdd.isEmpty() else 0
    avg_social = social_scores_rdd.mean() if not social_scores_rdd.isEmpty() else 0
    print("\nCâu 5: Điểm trung bình các bài thi tổ hợp:")
    print(f"- Bài thi Tự nhiên (Lí, Hóa, Sinh): {avg_natural:.2f} điểm")
    print(f"- Bài thi Xã hội (Sử, Địa, GDCD): {avg_social:.2f} điểm")

    # Câu 6 & 7: Vẽ biểu đồ
    print("\n--- Bắt đầu tạo biểu đồ (Câu 6-7) ---")
    natural_counts = natural_scores_rdd.map(lambda s: int(s)).countByValue()
    social_counts = social_scores_rdd.map(lambda s: int(s)).countByValue()
    score_levels = list(range(0, 31))
    natural_plot_counts = [natural_counts.get(s, 0) for s in score_levels]
    social_plot_counts = [social_counts.get(s, 0) for s in score_levels]
    plt.figure(figsize=(14, 7)); bar_width = 0.4; index = np.arange(len(score_levels))
    plt.bar(index - bar_width/2, natural_plot_counts, bar_width, label='Bài thi Tự nhiên', color='skyblue')
    plt.bar(index + bar_width/2, social_plot_counts, bar_width, label='Bài thi Xã hội', color='salmon')
    plt.xlabel('Mức điểm tổng'); plt.ylabel('Số lượng bài thi'); plt.title('Số lượng bài thi Tự nhiên và Xã hội theo mức điểm')
    plt.xticks(index, score_levels, rotation=90); plt.legend(); plt.tight_layout()
    plt.savefig('diem_to_hop.png')
    print("Câu 6: Biểu đồ 'diem_to_hop.png' đã được tạo.")

    math_idx, lit_idx, lang_idx = subject_names.index('Toan'), subject_names.index('Van'), subject_names.index('Ngoai_ngu')
    math_scores = parsed_rdd.map(lambda p: p[1][math_idx]).filter(lambda s: s is not None).collect()
    lit_scores = parsed_rdd.map(lambda p: p[1][lit_idx]).filter(lambda s: s is not None).collect()
    lang_scores = parsed_rdd.map(lambda p: p[1][lang_idx]).filter(lambda s: s is not None).collect()
    plt.figure(figsize=(10, 6))
    plt.boxplot([math_scores, lit_scores, lang_scores], tick_labels=['Toán', 'Văn', 'Ngoại ngữ'])
    plt.ylabel('Điểm'); plt.title('Thống kê điểm các môn Toán, Văn, Ngoại ngữ'); plt.grid(True)
    plt.savefig('boxplot_diem_cac_mon.png')
    print("Câu 7: Biểu đồ 'boxplot_diem_cac_mon.png' đã được tạo.")

    # Câu 8, 9, 10, 11
    print("\n--- Bắt đầu phân tích nâng cao và tạo biểu đồ (Câu 8-11) ---")
    province_tens = parsed_rdd.filter(lambda p: any(s == 10.0 for s in p[1] if s is not None)) \
                              .map(lambda p: (p[0][:2], 1)).reduceByKey(lambda a, b: a + b)
    if not province_tens.isEmpty():
        top_province = province_tens.max(key=lambda x: x[1])
        print(f"\nCâu 8: Tỉnh có nhiều thí sinh đạt điểm 10 nhất là tỉnh có mã '{top_province[0]}' với {top_province[1]} thí sinh.")

    province_avg_scores_data = parsed_rdd.map(lambda p: (p[0][:2], [s for s in p[1] if s is not None])) \
                                         .filter(lambda p: len(p[1]) > 0).map(lambda p: (p[0], (sum(p[1]), len(p[1])))) \
                                         .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                                         .map(lambda p: (p[0], p[1][0] / p[1][1])).sortByKey().collect()
    provinces = [item[0] for item in province_avg_scores_data]; avg_scores = [item[1] for item in province_avg_scores_data]
    plt.figure(figsize=(15, 8)); plt.bar(provinces, avg_scores, color='cornflowerblue')
    plt.xlabel('Mã Tỉnh'); plt.ylabel('Điểm trung bình'); plt.title('Điểm trung bình các môn thi theo Tỉnh')
    plt.xticks(rotation=90); plt.tight_layout()
    plt.savefig('diem_trung_binh_tinh.png')
    print("\nCâu 9: Biểu đồ 'diem_trung_binh_tinh.png' đã được tạo.")

    subject_map_by_name = {name: i for i, name in enumerate(subject_names)}
    blocks = {'A': ['Toan', 'Li', 'Hoa'], 'B': ['Toan', 'Hoa', 'Sinh'], 'C': ['Van', 'Su', 'Dia'], 'D': ['Toan', 'Van', 'Ngoai_ngu']}
    block_indices = {b: [subject_map_by_name.get(s) for s in subs] for b, subs in blocks.items()}
    def get_best_block(scores):
        s_block_scores = {}
        for b, inds in block_indices.items():
            if all(i is not None for i in inds):
                block_scores = [scores[i] for i in inds]
                if all(s is not None for s in block_scores): s_block_scores[b] = sum(block_scores)
        if not s_block_scores: return None
        max_score = max(s_block_scores.values())
        best_b = [b for b, s in s_block_scores.items() if s == max_score]
        chosen_b = 'A' if 'A' in best_b else best_b[0]
        return (chosen_b, round(max_score * 2) / 2)
    best_block_stats = parsed_rdd.map(lambda p: get_best_block(p[1])).filter(lambda x: x is not None).countByValue()

    stats_by_block = defaultdict(lambda: defaultdict(int))
    for (block, score), count in best_block_stats.items(): stats_by_block[block][score] = count
    score_range = np.arange(0, 30.5, 0.5); blocks_to_plot = ['A', 'B', 'C', 'D']
    block_counts = {block: [stats_by_block[block][s] for s in score_range] for block in blocks_to_plot}
    plt.figure(figsize=(18, 10)); bar_width = 0.2; index = np.arange(len(score_range))
    plt.bar(index - 1.5*bar_width, block_counts['A'], bar_width, label='Khối A', color='r')
    plt.bar(index - 0.5*bar_width, block_counts['B'], bar_width, label='Khối B', color='g')
    plt.bar(index + 0.5*bar_width, block_counts['C'], bar_width, label='Khối C', color='b')
    plt.bar(index + 1.5*bar_width, block_counts['D'], bar_width, label='Khối D', color='y')
    plt.xlabel('Mức điểm tổng'); plt.ylabel('Số lượng thí sinh'); plt.title('Thống kê mức điểm tổng theo khối thi cao nhất')
    plt.xticks(index[::2], score_range[::2]); plt.legend(); plt.tight_layout()
    plt.savefig('thong_ke_diem_khoi.png')
    print("Câu 11: Biểu đồ 'thong_ke_diem_khoi.png' đã được tạo.")

    print("\n--- Hoàn tất tất cả các yêu cầu ---")
    spark.stop()

if __name__ == "__main__":
    main()
