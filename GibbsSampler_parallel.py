from pyspark.sql import SparkSession
from pyspark import SparkContext
import numpy as np
from numpy.random import randint  # np.random.randint() [ )
from sampling import sample
import math
import time
import pickle
import socket

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("MotifDiscovery") \
    .getOrCreate()

# sc = SparkContext("spark://master:7077", "Gibbs")

def print_partition_info(index):
    # 获取当前节点的 IP 地址
    node_ip = socket.gethostbyname(socket.gethostname())
    print(f"分区 {index} 在节点 {node_ip} 上处理")
    return None


# 加载数据
with open('/root/sequences_with_motifs_100.pkl', 'rb') as file:
    sequences_with_motifs_100 = pickle.load(file)

with open('/root/label_starting_pos_100.pkl', 'rb') as file:
    label_starting_pos_100 = pickle.load(file)

with open('/root/label_motif_100.pkl', 'rb') as file:
    label_motif_100 = pickle.load(file)

# 初始化全局参数
group_nums = 1
K = 400  # Number of sequences in each group
N = 200  # Length of each sequence
alphabet = ['A', 'T', 'C', 'G']
b = 0.5  # 可调参参数
MAX_ITER = 50  # 最大迭代次数


def compute_model(sequences, pos, alphabet, w, b):
    q = {x: [b] * w for x in alphabet}
    p = {x: b for x in alphabet}

    for i in range(len(sequences)): # 1/3 * 399
        # print('pos:',pos)
        # print('type是：',type(pos))
        start_pos = pos[i]
        for j in range(w):
            c = sequences[i][start_pos + j]
            q[c][j] += 1

    for c in alphabet:
        for j in range(w):
            q[c][j] = q[c][j] / float(K - 1 + len(alphabet))

    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            if j < pos[i] or j > pos[i] + w:
                c = sequences[i][j]
                p[c] += 1

    total = sum(p.values())
    for c in alphabet:
        p[c] = p[c] / float(total)

    return q, p


def compute_F(sequences, pos, b, w, motif_counts_matrix, num_sequences, background_probs):
    q, p = compute_model(sequences, pos, alphabet, w, b)
    F = 0
    for i in range(w):
        for base in alphabet:
            q_ij = q[base][i]
            p_j = p[base]
            F += motif_counts_matrix[base][i] * math.log(q_ij / p_j)
    return F


# 每个节点计算权重矩阵的函数
def compute_weights(seq, partition , motif_len, b):
    node_sequences, node_pos = partition[0], partition[1]
    # print('node_seqs:', node_sequences)
    # print('node_pos', node_pos)
    q, p = compute_model(node_sequences, node_pos, alphabet, motif_len, b)

    weights = []
    qx = [1] * (N - motif_len + 1)  # ith seq = seq_x
    px = [1] * (N - motif_len + 1)  # ith sseq = seq_x
    for j in range(N - motif_len + 1):
        for k in range(motif_len):
            c = seq[j + k]
            qx[j] *= q[c][k]
            px[j] *= p[c]
    weights.append([x / y for (x, y) in zip(qx, px)])  # 计算比率
    return weights

# 并行处理399条序列
def process_sequence_in_parallel(seq, seq_minus, pos_minus, motif_len, spark):
    '''
    seq: seq_x, the exluded seq
    sequences: 1/3 * 399 seqs
    pos: 1/3 * 399 poss
    '''
    # rdd = spark.sparkContext.parallelize(list(zip(seq_minus, pos_minus)), 3)
    rdd = spark.sparkContext.parallelize(list(zip(seq_minus, pos_minus)), 3)
    # 在分区中提取出两个列表：一个存储所有元组的第一个元素 (seq)，另一个存储第二个元素 (pos)
    result_rdd = rdd.mapPartitions(lambda partition: [
        # 先将迭代器转换为列表
        partition_list := list(partition),
        # 提取 seq 和 pos
        [seq_pos[0] for seq_pos in partition_list],  # 提取 seq
        [seq_pos[1] for seq_pos in partition_list]  # 提取 pos
    ][1:])  # result_rdd 含有3个分区的数据，都储存在一个列表，第1个是分区1seq_list，第2个元素是分区1pos_list，第3个元素是分区2seq_list ...
    # print('result_rdd:', result_rdd.collect())

    # 计算每个分区的结果并生成 weights_rdd
    weights_rdd = result_rdd.mapPartitions(
                                 lambda partition: compute_weights(seq, partition, motif_len, b)
                                 )  # partition and result_rdd have inverse orders

    # 收集并打印结果
    weights_list = weights_rdd.collect()
    # print('weights_list', weights_list)

    # 合并权重矩阵
    combined_weights = list(np.mean(weights_list, axis=0))  # array to list

    return combined_weights  # not normalized  , list, len = N - w + 1


def compute_log_prob_ratio(F, K, possible_positions, weights):
    G = F
    for i in range(K):
        L_prime = possible_positions[i]
        G -= math.log(L_prime)
        for j in range(L_prime):
            Y_ij = weights[i][j]
            if Y_ij > 0:
                G -= Y_ij * math.log(Y_ij)
    return G


def compute_overlap_percentage(start1, len1, start2, len2):
    # starting pos and ending_pos of overlap
    start_overlap = max(start1, start2)
    end_overlap = min(start1 + len1, start2 + len2)

    # if overlap exists
    if start_overlap < end_overlap:
        overlap = end_overlap - start_overlap + 1
    else:
        overlap = 0
    return overlap / len1


# 主循环
start_time = time.time()

values = list(sequences_with_motifs_100.values())
F_ContainAllGroups_list = []
StartingPos_ContainAllGroups_list = []
MotifLen_ContainAllGroups_list = []
Overlap_ContainAllGroups_list = []

for group_num in range(group_nums):
    print('group:', group_num)
    sequences = values[group_num]
    G = -np.inf  # 初始化 G 值
    final_starting_pos_dict= {}
    F_ContainAllMotifLens_list = []

    for motif_len in range(5, 23):  # 尝试不同motif长度
        print('motif length: ', motif_len)
        pos = [randint(0, N - motif_len + 1) for _ in range(K)]  # 初始化起始位置   len = 400
        # print('初始化的pos：',pos)
        F = -np.inf
        F_list = []
        starting_pos_list = []

        for it in range(MAX_ITER):  # 迭代
            print('iter_num:', it)
            possible_positions = [N - motif_len + 1] * K

            for i in range(K):  # 选择第 i 条序列 x
                seq_minus = sequences[:]
                del seq_minus[i]  # 排除当前序列
                pos_minus = pos[:]
                del pos_minus[i]  # del starting_pos info of the ith seq   len = 399
                # print('去掉一个序列后的pos_minus：',pos_minus)
                # 并行处理399条序列
                Ai = process_sequence_in_parallel(sequences[i], seq_minus, pos_minus, motif_len, spark)
                                                 # ith条序列    399条序列   399个pos的list    value

                # 使用合并的权重矩阵更新序列 x 的起始位置
                norm_c = sum(Ai)
                Ai = [x / norm_c for x in Ai]  # normalized
                pos[i] = sample(range(N - motif_len + 1), Ai)

            # 继续执行motif和背景概率的计算
            motif_count_matrix = {x: [0] * motif_len for x in alphabet}
            for idx in range(len(sequences)):
                start_pos = pos[idx]
                for a in range(motif_len):
                    c = sequences[idx][start_pos + a]
                    motif_count_matrix[c][a] += 1

            background_probs = {x: 0 for x in alphabet}
            for idx in range(len(sequences)):
                for j in range(len(sequences[idx])):
                    if j < pos[idx] or j > pos[idx] + motif_len:
                        c = sequences[idx][j]
                        background_probs[c] += 1

            # 计算新的 F 值
            F_new = compute_F(sequences, pos, b, motif_len, motif_count_matrix, K, background_probs)
            if F_new > F:
                F = F_new
                pos_final = pos
            F_list.append(F_new)
        final_starting_pos_dict[motif_len] = pos_final
        # calculate G to determine the w
        # calculate normalized weights of complete alignment
        weights = []

        q_all, p_all = compute_model(sequences, pos, alphabet, motif_len, b)  # based on complete alignment
        for i in range(K):
            # Calculate probabilities for each possible position in sequence i
            qx = [1] * (N - motif_len + 1)  # the likelihood of each starting pos
            px = [1] * (N - motif_len + 1)  # the background likelihood of each starting pos
            for j in range(N - motif_len + 1):  # starting pos
                for k in range(motif_len):  # len of motif
                    c = sequences[i][j + k]
                    qx[j] *= q_all[c][k]  # Motif probability matrix
                    px[j] *= p_all[c]  # Background probability matrix
            # Compute the ratio between motif and background
            Ai = [x / y for (x, y) in zip(qx, px)]  # weight for each position
            norm_c = sum(Ai)
            Ai = list(map(lambda x: x / norm_c, Ai))  # Normalize to get probabilities
            weights.append(Ai)  # K * (N-w+1)
        G_new = compute_log_prob_ratio(F, K, possible_positions, weights) / (3 * motif_len)
        if G_new > G:
            G = G_new
            final_motif_len = motif_len
        F_ContainAllMotifLens_list.append(F_list)
    MotifLen_ContainAllGroups_list.append(final_motif_len)  # 10 * 1

    overlap_percentage = []  # store overlap percentage in each group
    for i in range(K):  # 400
        overlap = compute_overlap_percentage(label_starting_pos_100[group_num][i], len(label_motif_100[group_num][0]),
                                             final_starting_pos_dict[motif_len][i], final_motif_len)
        overlap_percentage.append(overlap)
    F_ContainAllGroups_list.append(F_ContainAllMotifLens_list)  # 10 * 18 * 100
    StartingPos_ContainAllGroups_list.append(final_starting_pos_dict)  # 10 * 18 * 400
    Overlap_ContainAllGroups_list.append(overlap_percentage)  # 10 *400

end_time = time.time()
elapsed_time = end_time - start_time
print('operation time: ', elapsed_time)

with open('./F.pkl', 'wb') as file:
    pickle.dump(F_ContainAllGroups_list, file)

with open('./StartingPos.pkl', 'wb') as file:
    pickle.dump(StartingPos_ContainAllGroups_list, file)

with open('./MotifLen.pkl', 'wb') as file:
    pickle.dump(MotifLen_ContainAllGroups_list, file)

with open('./Overlap.pkl', 'wb') as file:
    pickle.dump(Overlap_ContainAllGroups_list, file)

# 关闭 SparkSession
spark.stop()
