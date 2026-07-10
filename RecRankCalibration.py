import torch.nn as nn
import torch

class RecRankCalibration(nn.Module):
    def __init__(self):
        pass

    def forward(self, task_relu, task_scores):
        """
        仅说明task_relu为[batch_size, bucket_size], task_scores为[batch_size, 1]的情况
        """
        batch_size, bucket_size = task_relu.shape
        outputs = torch.zeros(batch_size, 1)
        # 对每个batch
        for i in range(batch_size):
            cur_task_relu = task_relu[i]  # 当前batch 桶的权重序列
            cur_task_score = task_scores[i] # task_scores[i]在[0, 1)中间
            bucket_level = cur_task_score * bucket_size  # 计算当前任务的分数对应的桶级别
            int_bucket_level = int(bucket_level)
            decimal_part = bucket_level - int_bucket_level  # 计算整数和小数部分

            cali = 0  # 开始计算校准后的分数
            for j in range(int_bucket_level):
                cali += cur_task_relu[j]
            cali += cur_task_relu[i][int_bucket_level] * decimal_part  # 加上小数部分对应的分数（最后一个桶未满，按照剩余的小数部分去乘该桶的权重）
            cali /= bucket_size  # normalize

            outputs[i, 0] = torch.log(cali / (1.0 - cali))  # 将校准后的分数转换为logit形式
        return outputs