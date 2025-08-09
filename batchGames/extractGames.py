import re,os
path='0803/gameGened300.txt'
with open(path, 'r') as f:
    content = f.read()

pattern=r'</think>\n(.*?)\n(?=Prompt: <|im_start|>system|$)'
matches = re.findall(pattern, content, re.DOTALL)
def extract(text):
    text = text.strip().replace('```python', '').replace('```', '')
    return text

extracted_texts = [extract(match) for match in matches if match.strip()]
namePattern=r'class\s+(\w+)\(BoardGame'
for i, text in enumerate(extracted_texts[:1000]):
    class_name = re.search(namePattern, text)
    if class_name:
        class_name = class_name.group(1)
        basePath=path=f'0803/games/{class_name}.py'
        count=1
        while os.path.exists(path):
            count += 1
            path = f'0803/games/{class_name}_{count}.py'
        if count>1:
            text=text.replace(f'class {class_name}(Board', f'class {class_name}_{count}(Board')
        with open(path, 'w') as f:
            f.write(text)
