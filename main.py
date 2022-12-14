import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tabulate import tabulate

def get_attention_head(model, layer_num, head_num):
    attn = model.transformer.h[layer_num].attn
    embed_dim, num_heads, head_dim = attn.embed_dim, attn.num_heads, attn.head_dim

    c_attn = attn.c_attn.weight.detach()
    WQ, WK, WV = c_attn.reshape(embed_dim, 3, embed_dim).permute(1, 0, 2)
    WQ_head = WQ.reshape(embed_dim, num_heads, head_dim).permute(1, 0, 2)[head_num]
    WK_head = WK.reshape(embed_dim, num_heads, head_dim).permute(1, 0, 2)[head_num]
    WV_head = WV.reshape(embed_dim, num_heads, head_dim).permute(1, 0, 2)[head_num]

    WO = attn.c_proj.weight.detach()
    WO_head = WO.reshape(num_heads, head_dim, embed_dim)[head_num]

    return WQ_head, WK_head, WV_head, WO_head

def create_svd_table(tokenizer, WE, WO_head, WV_head, num_rows, num_columns):
    U, S, V = torch.linalg.svd(WV_head @ WO_head)
    tokens = []

    for i in range(num_columns):
        logits = WE @ V[i, :]
        ids = torch.topk(logits, k=num_rows).indices
        tokens.append(tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True))

    print(tabulate(list(zip(*tokens))))

layer_num = 22
head_num = 10
num_rows = 20
num_columns = 7

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

WE = model.transformer.wte.weight.detach()
WQ_head, WK_head, WV_head, WO_head = get_attention_head(model, layer_num, head_num)

print(f"Decoded SVD columns from layer {layer_num}, head {head_num}:")
create_svd_table(tokenizer, WE, WO_head, WV_head, num_rows, num_columns)

