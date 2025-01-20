import torch
import random
from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')

text = '수학 문제를 잘 풀기 위해서는'

input_ids = tokenizer.encode(sent, return_tensors='pt')
print(input_ids)

output = model.generate(input_ids,
                        max_length=128,
                        repetition_penalty=2.0,
                        use_cache=True)
output_ids = output.numpy().tolist()[0]
print(output_ids)

tokenizer.decode(output_ids)

output = model(input_ids)

# logits.shape == torch.Size([51200]). 즉, 총 단어 집합 크기만큼의 차원을 가지는 벡터.
logits = output.logits[0, -1]

top5 = torch.topk(logits, k=5)
tokens = [tokenizer.decode(token_id) for token_id in top5.indices.tolist()]
print(tokens)

text = '수학 문제를 잘 풀기 위해서는'
input_ids = tokenizer.encode(text, return_tensors='pt')

while len(input_ids[0]) < 50:
    with torch.no_grad():
        output = model(input_ids)
    logits = output.logits[0, -1]
    top5 = torch.topk(logits, k=30)
    token_id = random.choice(top5.indices.tolist())
    input_ids = torch.cat([input_ids, torch.tensor([[token_id]])], dim=1)

tokenizer.decode(input_ids[0])
