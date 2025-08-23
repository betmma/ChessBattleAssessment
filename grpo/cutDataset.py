# keep N rows per game for top k games with most rows
import json, collections

file='../evaluation_results_vllm/grpo/DrCoNi_lv3_raw_d6.jsonl'
newFile='../evaluation_results_vllm/grpo/DrCoNi_lv3_raw_d6_balanced.jsonl'
k=3
N=536
#######
with open(file, 'r') as f:
    data = [json.loads(line) for line in f]
    
data2,states=[],[]
for entry in data:
    state=entry['task']+entry['prompt'][1]['content']
    if state not in states:
        states.append(state)
        data2.append(entry)
data=data2

tasks=[entry['task'] for entry in data]
taskCount=collections.Counter(tasks)

newData=[]
for task, count in taskCount.most_common(k):
    print(f"Task: {task}, Count: {count}")
    taskEntries = [entry for entry in data if entry['task'] == task]
    newData.extend(taskEntries[:N])
print(f"Total entries after filtering: {len(newData)}")

with open(newFile, 'w') as f_out:
    for entry in newData:
        f_out.write(json.dumps(entry,ensure_ascii=False) + '\n')
print(f"Filtered dataset saved to {newFile}")