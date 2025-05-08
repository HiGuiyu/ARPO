import numpy as np
import math
import json
import random
from collections import defaultdict

class ReplayBuffer():

    def __init__(self, json_path, n_rollout):
        self.n_rollout = n_rollout

        with open(json_path, 'r') as f:
            replay_data = json.load(f)
        
        self.dataset = defaultdict(list)
        
        for data in replay_data:
            # task_id, history_images, history_messages, eval_result
            task_id = data['task_id']
            self.dataset[task_id].append(data)
    
    def update_replay_buffer(self, task_config, history_messages, eval_result):
        task_id = task_config["task_id"]
        task_replay_buffer = self.dataset[task_id]
        
        task_replay_buffer.append(
            {
                "task_id": task_config["task_id"],
                "domain": task_config["domain"],
                "instruction": task_config['instruction'],
                "history_messages": history_messages,
                "eval_result": eval_result,
            }
        )
        if len(task_replay_buffer) > self.n_rollout:
            task_replay_buffer.pop(0)

    def update_replay_buffer_batch(self, task_configs, history_messages_batch, eval_results_batch):
        for task_config, history_messages, eval_result in zip(task_configs, history_messages_batch, eval_results_batch):
            self.update_replay_buffer(task_config, history_messages, eval_result)
        

    def get_batch2(self, task_configs, eval_results):

        task_id_set = set([task_config['task_id'] for task_config in task_configs])

        history_messages_replay, eval_results_replay, task_configs_replay = [], [], []
        for task_id in task_id_set:
            task_config = [task_config for task_config in task_configs if task_config['task_id'] == task_id][0]
        
            eval_results_in_group = []
            
            for task_config, eval_result in zip(task_configs, eval_results):
                if task_config['task_id'] == task_id:
                    eval_results_in_group.append(eval_result)
            
            eval_results_in_group = np.array(eval_results_in_group)
            num_success = (eval_results_in_group >= 0.5).sum()
            num_fail = (eval_results_in_group < 0.5).sum()
            num_total = len(eval_results_in_group)

            candidates = self.dataset[task_id]
            # TODO: complete the rest

        return task_configs_replay, history_messages_replay, eval_results_replay

    def get_batch(self, task_configs, eval_results):
        """
        对于输入的当前 batch（包含多个任务组），
        从 replay buffer 中为每个任务 group 采样相同数量的数据（扩充一倍），
        并尽量使得扩充后整体（当前数据+replay数据）的 success 的比例在 0.2~0.8 之间。
        """
        history_messages_replay = []
        eval_results_replay = []
        task_configs_replay = []
        
        # 获取所有任务 id（即 group）
        unique_task_ids = set(tc['task_id'] for tc in task_configs)
        
        # 对每个任务 id 分别采样 replay 数据
        for task_id in unique_task_ids:
            # 找出当前 batch 中此任务对应的所有样本的索引
            cur_task_config = [tc for tc in task_configs if tc['task_id'] == task_id][0]

            group_indices = [i for i, tc in enumerate(task_configs) if tc['task_id'] == task_id]
            n_group = len(group_indices)
            
            # 统计当前组中成功的样本数（eval_result >= 0.5 视为成功）
            current_success = sum(1 for i in group_indices if eval_results[i] >= 0.5)
            
            # 计算 replay 中需要采样的成功样本数的允许区间
            lower_bound = max(0, int(math.ceil(0.4 * n_group - current_success)))
            upper_bound = min(n_group, int(math.floor(1.6 * n_group - current_success)))
            
            # 选择目标成功数。如果区间无效则直接取 n/2，否则取 round(n/2) 并 clamp 到 [lower_bound, upper_bound]
            if lower_bound > upper_bound:
                target_success = n_group // 2
            else:
                target_success = round(n_group / 2)
                target_success = min(max(target_success, lower_bound), upper_bound)
            target_fail = n_group - target_success
            
            # 从 replay buffer 中提取候选样本，注意 candidates 为该任务的所有历史数据
            candidates = self.dataset[task_id]
            # 将候选按照成功和失败分开
            replay_successes = [item for item in candidates if item['eval_result'] >= 0.5]
            replay_fails = [item for item in candidates if item['eval_result'] < 0.5]
            
            # 定义辅助函数：尝试采样 count 个样本，优先无放回采样，不足时采用放回采样
            def sample_items(items, count):
                if count <= 0:
                    return []
                if len(items) >= count:
                    return random.sample(items, count)
                else:
                    return items + random.choices(items, k=count - len(items))
            
            # 如果需要某一类样本，但候选中该类为空，则直接从所有候选中抽取
            if (target_success > 0 and len(replay_successes) == 0) or (target_fail > 0 and len(replay_fails) == 0):
                selected = sample_items(candidates, n_group)
            else:
                selected_successes = sample_items(replay_successes, target_success)
                selected_fails = sample_items(replay_fails, target_fail)
                selected = selected_successes + selected_fails
                # 如果合计不足 n_group，则补充
                if len(selected) < n_group:
                    selected += sample_items(candidates, n_group - len(selected))
                # 如果多于 n_group，则截断
                if len(selected) > n_group:
                    selected = selected[:n_group]
                random.shuffle(selected)
            
            # 将采样到的 replay 数据中的 task_config, history_messages, eval_result 添加到返回结果中
            for item in selected:
                task_configs_replay.append(cur_task_config)
                history_messages_replay.append(item["history_messages"])
                eval_results_replay.append(item["eval_result"])
        
        return task_configs_replay, history_messages_replay, eval_results_replay
