import json, collections

file='../evaluation_results_vllm/grpo/DrCoNi_1000.jsonl'
with open(file, 'r') as f:
    data = [json.loads(line) for line in f]
print(f"Total entries: {len(data)}")
data2,states=[],[]
for entry in data:
    state=entry['task']+entry['prompt'][1]['content']
    if state not in states:
        states.append(state)
        data2.append(entry)
data=data2
print(f"Unique states: {len(states)}")
tasks=[entry['task'] for entry in data]
taskCount=collections.Counter(tasks)
print(f"Tasks: {taskCount}")
rewards = [entry['reward_model']['ground_truth'].values() for entry in data]
ave=lambda x: sum(x)/len(x) if x else 0
print(f'Average reward: {ave([ave(r) for r in rewards])}')
print(f'Max reward: {ave([max(r) for r in rewards])}')
print(f'Min reward: {ave([min(r) for r in rewards])}')