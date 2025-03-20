import google.generativeai as genai

genai.configure(api_key="AIzaSyAF7SLJ_ldhzLOYE-yAZeagEq72eTOmO_k")
models = genai.list_models()

for model in models:
    print(model.name)
