import json, os, sys, multiprocessing as mp
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from games import GameByName
from agents.universal_minimax_agent import UniversalMinimaxAgent
from utils.safe_json_dump import clean_np_types

_MINIMAX_AGENT=None; _MAX_DEPTH=None; GAMES_FOLDER=''

def _init_minimax_worker(depth:int):
    global _MINIMAX_AGENT,_MAX_DEPTH; _MAX_DEPTH=depth; _MINIMAX_AGENT=UniversalMinimaxAgent(max_depth=depth) if depth>0 else None

def filter_rewards(r):
    v=list(r.values()); return len(v)>1 and (any(x>990 for x in v) or any(x<-990 for x in v)) and not all(x==v[0] for x in v)

def _iter_moves(detailed_logs):
    for g,gd in detailed_logs.items():
        for gr in gd.get('games',{}).values():
            for mv in gr.get('moves',[]):
                yield g,mv

def _compute_entry_for_board(task):
    g,b=task
    if not (_MAX_DEPTH and _MAX_DEPTH>0): return None
    game=GameByName(g,GAMES_FOLDER)()
    try: game.load_state_from_representation(b)
    except Exception as e:
        print(e);return None
    agent=_MINIMAX_AGENT or UniversalMinimaxAgent(max_depth=_MAX_DEPTH)
    try: 
        ar=agent.get_action_rewards(game)
    except Exception as e:
        print(e);return None
    if not filter_rewards(ar): return None
    return {"prompt":game.get_chat_history_for_llm(agent),"task":g,"reward_model":{"ground_truth":ar}}

def _extract_board_from_prompt(prompt):
    for m in prompt:
        if m.get('role')=='user':
            c=m.get('content','')
            if 'You are player' in c: return c.split('You are player')[0].strip()
    return None

def generate_dataset(input_file:str, output_file:str, max_depth:int=4):
    data=json.load(open(input_file)); detailed=data.get('detailed_logs',{})
    processed_boards=set(); out=open(output_file,'w'); depth=max_depth
    if depth<=0:  # sequential path (uses existing move rewards)
        total=sum(len(gr.get('moves',[])) for gd in detailed.values() for gr in gd.get('games',{}).values())
        gen=0
        with tqdm(total=total,desc='Processing moves',unit='move') as pbar:
            for g,mv in _iter_moves(detailed):
                pbar.update(1)
                b=mv.get('board_before','')
                if not b: continue
                k=f'{g}:{b}'
                if k in processed_boards: continue
                processed_boards.add(k)
                game=GameByName(g,GAMES_FOLDER)(); game.load_state_from_representation(b)
                ar=mv.get('action_rewards',{})
                if not ar or not filter_rewards(ar): continue
                prompts=game.get_chat_history_for_llm(UniversalMinimaxAgent(max_depth=depth))
                out.write(json.dumps({"prompt":prompts,"task":g,"reward_model":{"ground_truth":ar}})+'\n'); gen+=1
                pbar.set_description(f'Processing moves (Generated: {gen} entries)')
        print(f"\nTotal unique board states processed: {gen}"); out.close(); return
    # parallel path: build unique tasks
    tasks=[]
    for g,mv in _iter_moves(detailed):
        b=mv.get('board_before','')
        if not b: continue
        k=f'{g}:{b}'
        if k in processed_boards: continue
        processed_boards.add(k); tasks.append((g,b))
    gen=skip=0; workers=max(1,(os.cpu_count() or 1))
    with tqdm(total=len(tasks),desc='Processing boards',unit='board') as pbar, mp.Pool(processes=workers,initializer=_init_minimax_worker,initargs=(depth,)) as pool:
        for res in pool.imap_unordered(_compute_entry_for_board,tasks,chunksize=16):
            pbar.update(1)
            if res is None: skip+=1; continue
            out.write(json.dumps(clean_np_types(res))+'\n'); gen+=1
            pbar.set_description(f'Processing boards (Generated: {gen}, Skipped: {skip})')
    out.close(); print(f"\nTotal unique board states processed: {gen}")

def recalculate_rewards_from_jsonl(input_jsonl:str, output_jsonl:str, new_max_depth:int=6):
    agent=UniversalMinimaxAgent(max_depth=new_max_depth)
    total=sum(1 for _ in open(input_jsonl))
    processed=skip=0
    with open(input_jsonl) as fin, open(output_jsonl,'w') as fout, tqdm(total=total,desc='Recalculating rewards',unit='entry') as pbar:
        for line in fin:
            pbar.update(1)
            e=json.loads(line.strip())
            task=e.get('task',''); prompt=e.get('prompt',[])
            if not prompt: skip+=1; continue
            b=_extract_board_from_prompt(prompt)
            if not b: skip+=1; continue
            game=GameByName(task,GAMES_FOLDER)(); game.load_state_from_representation(b)
            ar=agent.get_action_rewards(game)
            if not filter_rewards(ar): skip+=1; continue
            e['reward_model']['ground_truth']=ar; fout.write(json.dumps(e)+'\n'); processed+=1
            pbar.set_description(f'Recalculating rewards (Processed: {processed}, Skipped: {skip})')
    print(f"\nRecalculation complete!\nTotal entries processed: {processed}\nTotal entries skipped: {skip}\nUpdated dataset saved to: {output_jsonl}")

if __name__=='__main__':
    input_file='/remote-home1/yrmou/ChessBattleAssessment/evaluation_results_vllm/game_logs/20250829-020510_API Agent (Qwen3-8B)_vs_API Agent (Qwen3-8B)_CONSOLIDATED.json'
    max_depth=8
    output_file=f'evaluation_results_vllm/grpo/8bgames_depth{max_depth}.jsonl'
    GAMES_FOLDER='/remote-home1/yrmou/ChessBattleAssessment/debug/evo_games_20250827_081440/all_success_unique'
    if not os.path.exists(input_file): print(f'Error: Input file {input_file} does not exist'); sys.exit(1)
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    if input_file.endswith('.json'):
        generate_dataset(input_file, output_file,max_depth)
    else:
        recalculate_rewards_from_jsonl(input_file, output_file, new_max_depth=max_depth)
    print(f'Dataset generated successfully at {output_file}')
