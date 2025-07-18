import json

file='evaluation_results_vllm/grpo/8games_8b_battle_depth4.jsonl'
with open(file, 'r') as f:
    data = [json.loads(line) for line in f]
rewards = [entry['reward_model']['ground_truth'].values() for entry in data]
print(f"Total entries: {len(data)}")
ave=lambda x: sum(x)/len(x) if x else 0
print(f'Average reward: {ave([ave(r) for r in rewards])}')
print(f'Max reward: {ave([max(r) for r in rewards])}')
print(f'Min reward: {ave([min(r) for r in rewards])}')