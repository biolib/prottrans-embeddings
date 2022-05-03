import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
from tqdm import tqdm
import argparse
import io
from Bio import SeqIO
import gc

print("Step 1/2 | Loading transformer model...")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

parser = argparse.ArgumentParser()
parser.add_argument('--fasta')
args = parser.parse_args()

input_str = open(args.fasta, "r").read()

if not input_str.strip().startswith(">") and input_str.strip() != "":
    input_str = "> Unnamed Protein \n" + input_str

input_str_buffer = io.StringIO(input_str)

sequences = []
for idx, record in enumerate(SeqIO.parse(input_str_buffer, "fasta")):
    sequences.append(str(record.seq))

fasta_len = len(sequences)

print("\nStep 2/2 | Generating embeddings for sequences...")
sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
input_ids = torch.tensor(ids['input_ids']).to(device)
attention_mask = torch.tensor(ids['attention_mask']).to(device)

with torch.no_grad():
    embedding = model(input_ids=input_ids,attention_mask=attention_mask)
embedding = embedding.last_hidden_state.cpu().numpy()

features = [] 
for seq_num in range(len(embedding)):
    seq_len = (attention_mask[seq_num] == 1).sum()
    seq_emd = embedding[seq_num][:seq_len-1]
    features.append(seq_emd)
    
print(features)