import sys, json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

label, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
MAXLEN, STRIDE = 1024, 512
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map={'':'cuda:0'}); model.eval()
tok = AutoTokenizer.from_pretrained(path)
ds = load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test')
text = "\n\n".join(t for t in ds['text'] if len(t.strip())>0)
ids_all = tok(text, return_tensors='pt', add_special_tokens=False)['input_ids']
seq_len = ids_all.shape[1]
nlls, n_tok, prev_end = 0.0, 0, 0
for begin in range(0, seq_len, STRIDE):
    end = min(begin+MAXLEN, seq_len)
    trg = end - prev_end
    ids = ids_all[:, begin:end].to('cuda:0')
    tgt = ids.clone(); tgt[:, :-trg] = -100
    with torch.no_grad():
        loss = model(ids, labels=tgt).loss
    nlls += loss.item()*trg; n_tok += trg; prev_end = end
    if end == seq_len: break
ppl = float(np.exp(nlls/n_tok))
json.dump({'label':label,'ppl':ppl,'n_tokens':n_tok,'seq_len':seq_len,'method':'rolling_max1024_stride512_masked'}, open(out,'w'))
print(f'{label}: rolling-PPL={ppl:.2f} (scored {n_tok}/{seq_len} tokens)')
