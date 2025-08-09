import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

modelpath='/mnt/data/user/zhao_jun/mou_yu_rong/openrlhf/chessBattleAdvanced/Qwen3-32B' # 'Qwen3-8B
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelpath)

# Configurae the sampling parameters (for thinking mode)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384)

# Initialize the vLLM engine
llm = LLM(model=modelpath,max_model_len=16384)
def gen(systemPrompt,userPrompt,thinking=True):
    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": userPrompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,  # Set to False to strictly disable thinking
    )
    print('After applying chat template:\n----------------------\n',text)
    outputs = llm.generate([text], sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:\n----------------------\n{prompt!r}\nGenerated text:\n----------------------\n{generated_text!r}")

def batchGen(systemPrompt,userPrompts,thinking=True):
    texts = []
    for userPrompt in userPrompts:
        messages = [
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": userPrompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,  # Set to False to strictly disable thinking
        )
        texts.append(text)
    outputs = llm.generate(texts, sampling_params)
    outputTexts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        outputTexts.append((prompt, generated_text))
    return outputTexts

with open('board_game.py', 'r') as f:
    gameClass = f.read()

systemPrompt = "You are a game designer and expert programmer."
userPrompt = "Write a game in Python. The game should be a subclass of the BoardGame class. BoardGame class is as below:\n```\n" + gameClass + "\n```\nThe game should be BRAND NEW and UNIQUE. The game should not be similar to TicTacToe. there should be a simple way to assess board position. the game must be perfect information. average legal moves per state should be around 10, and average game should end in 20 steps. The board should contain all information (for example you cannot add a variable to track score, except that you write it into the board). Your output should only include the class of your game, beginning with:\n```\nimport numpy as np\nfrom typing import List, Any, Optional, Tuple\nfrom games.board_game import BoardGame\n```\ndo not repeat BoardGame class.\na random word to help you brainstorm: "
wordsPath='1krdwords.txt'
with open(wordsPath, 'r') as f:
    words = f.read().splitlines()

import random
words=random.sample(words, 300)  # test with 10 words
userPrompts = [userPrompt+word for word in words]
outputTexts = batchGen(systemPrompt, userPrompts, thinking=True)
with open('gameGened300.txt', 'w') as f:
    for prompt, generated_text in outputTexts:
        f.write(f"Prompt: {prompt}\nGenerated Text: {generated_text}\n\n")

