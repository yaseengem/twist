import ollama

response = ollama.generate(model='llama3.2:1b', prompt='What is the color of sky? Be specific.', stream=True)
for res in response:
    print(res['response'], end='')