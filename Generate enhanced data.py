from llmtuner.chat import ChatModel
from llmtuner.extras.misc import torch_gc
import pandas as pd
import sys
import random
sys.stdout.reconfigure(encoding='utf-8')

# df = pd.read_csv('instruction.csv',encoding='latin1')
# movie_names = df.iloc[:, 0]
# df1 = pd.read_csv('movie.csv', encoding='latin1')

# df1 = pd.read_csv('book_inter_instructions.csv', encoding='latin1')
# df2 = pd.read_csv('book.csv', encoding='latin1')
# # df1 = pd.read_csv('bookpro_instructions.csv',encoding='latin1')
# movie_names = df1.iloc[:, 0]

args = dict(
  model_name_or_path="your_path", # 使用 Llama-3-8b-Instruct 模型
  #adapter_name_or_path="8b-lora",            # 加载之前保存的 LoRA 适配器
  template="llama3",                     # 和训练保持一致
  finetuning_type="lora",                  # 和训练保持一致
  quantization_bit=4,                    # 加载 4 比特量化模型
  #use_unsloth=True,                     # 使用 UnslothAI 的 LoRA 优化来获得两倍的推理速度
)

chat_model = ChatModel(args)
k = 0
data = []
messages = []
print(" `clear` ， `exit` ")
selected = []
random.seed(42)
for x in movie_names:
    if x.strip() == "exit":
      break
    user_list = df2['instruction'].tolist()
    indices = list(range(len(user_list)))
    i = 0
    j = 0
    a = 0
    y = 0
    while indices:
        query = ''
        random_index = random.choice(indices)
        query = user_list[random_index]+x
        print(query)
        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)
        response = ""
        for new_text in chat_model.stream_chat(messages):
          print(new_text, end="", flush=True)
          response += new_text
        print()
        indices.remove(random_index)
        messages.append({"role": "assistant", "content": response})
        # datas = f'{response}'
        # data.append(datas)
        # k = k + 1
        # if (k % 5 == 0):
        #   print(k)
        messages = []
        torch_gc()
        print("对话历史已清除")
          # continue
        if response == 'Yes.':
            i = 1
            a = a+1
            if a == 1:
                selected.append(random_index + 2)
                data.append(k + 2)
        if response == 'No.':
            j = 1
            y = y+1
            if y == 1:
                selected.append(random_index + 2)
                data.append(k + 2)
        if i+j == 2:
          break
    k = k+1


# for i in movie_names:
#   query = i
#   if query.strip() == "exit":
#     break
#   messages.append({"role": "user", "content": query})
#   print("Assistant: ", end="", flush=True)
#
#   response = ""
#   for new_text in chat_model.stream_chat(messages):
#     print(new_text, end="", flush=True)
#     response += new_text
#   print()
#   messages.append({"role": "assistant", "content": response})
#   datas = f'{response}'
#   data.append(datas)
#   k = k+1
#   if (k % 10 == 0):
#     print(k)
#     messages = []
#     torch_gc()
#     print("Dialogue history cleared")
#     continue

# df_instructions = pd.DataFrame({'item_index': data, 'user_index': selected})
# # df_instructions = pd.DataFrame({'genres': data})
# df_instructions.to_csv('book_inter_result.csv', index=False)
# torch_gc()