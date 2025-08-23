import json, collections

file='../evaluation_results_vllm/grpo/DrCoNi_lv2_raw2_balanced.jsonl'
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
def rewardStatistics(rewards):
    aveave=ave([ave(r) for r in rewards])
    maxave=ave([max(r) for r in rewards])
    minave=ave([min(r) for r in rewards])
    print(f"Average reward: {aveave}")
    print(f"Max reward: {maxave}")
    print(f"Min reward: {minave}")
    specialRewards={999:'Win in 1',-999:'Lose in 1',998:'Win in 1',-998:'Lose in 1',997:'Win in 2',-997:'Lose in 2',996:'Win in 2',-996:'Lose in 2'}
    for reward,description in specialRewards.items():
        count = sum(1 for r in rewards if reward in r)
        if count==0:continue
        print(f"{description} ({reward}): {count} occurrences, {count/len(rewards)*100:.2f}% of total")
rewardStatistics(rewards)
# for each task print average, max, min reward
for task in taskCount:
    task_rewards = [entry['reward_model']['ground_truth'].values() for entry in data if entry['task'] == task]
    print(f"Task: {task}")
    rewardStatistics(task_rewards)