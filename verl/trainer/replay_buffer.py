import numpy as np
import json
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
    
    def update_replay_buffer(self, task_config, history_images, history_messages, eval_result):
        self.dataset[task_id].append(
            {
                "task_id": task_config["task_id"],
                "domain": task_config["domain"],
                "instruction": task_config['instruction'],
                "history_images": history_images,
                "history_messages": history_messages,
                "eval_result": eval_result,
            }
        )

    def update_replay_buffer_batch(self, task_configs, history_images_batch, history_messages_batch, eval_results_batch):
        for task_config, history_images, history_messages, eval_result in zip(task_configs, history_images_batch, history_messages_batch, eval_results_batch):
            self.update_replay_buffer(task_config, history_images, history_messages, eval_result)
        

    def get_batch(self, task_configs):

        task_configs_batch = []
        history_messages_batch = []
        eval_results_norm_batch = []
        for task_config in task_configs:
            # for each task, get n_rollout samples [n-1 history, 1 current]    
            task_id = task_config['id']

            datalist = self.dataset[task_id].copy()

            if len(datalist) < self.n_rollout:
                num_pad = self.n_rollout - len(datalist)
                datalist = datalist + datalist[-num_pad:]
            
            results = np.array([data['eval_result'] for data in datalist])
            mean = np.mean(results)
            std = np.std(results)

            results_norm = (results - mean) / (std + 1e-6)

            messages = [x["history_messages"] for x in datalist][-self.n_rollout:]
            results_norm = results_norm[-self.n_rollout:]
            print(f"task_id: {task_id}, num_samples: {len(datalist)}, messages: {len(messages)}")

            history_messages_batch.extend(messages)
            eval_results_norm_batch.extend(results_norm.tolist())
            task_configs_batch.extend([task_config] * len(results_norm))
        
        return task_configs_batch, history_messages_batch, eval_results_norm_batch



