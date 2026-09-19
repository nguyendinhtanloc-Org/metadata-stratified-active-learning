"""
Metric tính Shannon Entropy cho metadata stratification.
Đo lường mức độ đa dạng/đồng nhất của batch dữ liệu.
"""

import math
from collections import Counter

def calc_entropy(counter):
    """
    Tính Shannon Entropy từ Counter.
    
    Input:  Counter object, ví dụ Counter({"clear": 3, "rainy": 2, "foggy": 1})
    Output: float - giá trị entropy (bits)
    
    Công thức: H = -Σ p(x) * log2(p(x))
    - Entropy cao = phân bố đều = batch đa dạng
    - Entropy thấp = phân bố lệch = batch đồng nhất
    - Entropy = 0 = chỉ có 1 loại = batch hoàn toàn đồng nhất
    """

    total = sum(counter.values())
    entropy = 0.0

    for count in counter.values():
        p = count / total
        
        if p > 0: # Tránh log2(0) = -inf
            entropy -= p * math.log2(p)

    return entropy

def calc_batch_entropy(metadata_list, field):
    """
    Tính entropy của một batch dữ liệu trên một metadata field.
    
    Input:
        metadata_list: list of dicts từ extract_metadata()
            Ví dụ: [{"filename": "xxx.jpg", "weather": "clear", ...}, ...]
        field: tên field cần tính entropy
            Choices: "weather", "scene", "timeofday"
    
    Output: float - entropy của batch trên field đó
    """

    values = []

    # Trích xuất giá trị field từ tất cả records
    for record in metadata_list:
        values.append(record[field])

    # Đếm tần suất
    counter = Counter(values)

    # Tính entropy
    entropy = calc_entropy(counter)

    return entropy
