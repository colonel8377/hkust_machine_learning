import fp_growth_py3 as fpg

# 数据集
dataset = [
    ['C', 'D'],
    ['B', 'E', 'F'],
    ['A', 'C', 'D'],
    ['A', 'C', 'D'],
    ['C', 'D'],
    ['B', 'E', 'F'],
    ['A', 'C', 'D'],
    ['A', 'C', 'D'],
]

if __name__ == '__main__':
    frequent_itemsets = fpg.find_frequent_itemsets(dataset,
                                                   minimum_support=3,
                                                   include_support=True)
    print(type(frequent_itemsets))  # print type

    result = []
    for itemset, support in frequent_itemsets:  # 将generator结果存入list
        result.append((itemset, support))

    result = sorted(result, key=lambda i: i[0])  # 排序后输出
    for itemset, support in result:
        print(str(itemset) + ' ' + str(support))
