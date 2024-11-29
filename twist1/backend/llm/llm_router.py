import ollama


response = ollama.generate(model='codegemma', prompt='Write a Python program to draw a circle spiral in three colors')
print(response['response'])