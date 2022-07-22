import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
from tqdm import tqdm
import argparse
import io
from Bio import SeqIO
import gc
import zipfile
import time


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
processed_sequences = sequences.copy()
processed_sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
processed_sequences = [' '.join(list(seq)) for seq in sequences]

def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]
   
fh = open('embeddings.npz', 'wb')

proc_seq_chunks = chunks(processed_sequences, 100)
seq_idx = 0
chunk_idx = 1

print("Step 1/2 | Loading transformer model...")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
gc.collect()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()
with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for chunk in proc_seq_chunks:
        with torch.no_grad():
            for idx, proc_seq in enumerate(chunk):
                # if seq_idx < 3107:
                #     print(seq_idx)
                #     seq_idx += 1
                #     continue
                print(proc_seq.replace(' ', '')[:10])
                ids = tokenizer.batch_encode_plus([proc_seq], add_special_tokens=True, padding="longest")
                input_ids = torch.tensor(ids['input_ids']).to(device)
                attention_mask = torch.tensor(ids['attention_mask']).to(device)
                print(f"Generating embedding for seq of len {len(proc_seq.replace(' ', ''))} and idx {seq_idx} of {fasta_len}")

                embedding = model(input_ids=input_ids,attention_mask=attention_mask)

                embedding = embedding.last_hidden_state.cpu().numpy()

                for seq_num in range(len(embedding)):
                    seq_len = (attention_mask[seq_num] == 1).sum()
                    seq_emd = embedding[seq_num][:seq_len-1]
                    with zf.open(sequences[seq_idx] + '.npy', 'w', force_zip64=True) as buf:
                            np.lib.npyio.format.write_array(buf,
                                                            np.asanyarray(seq_emd),
                                                            allow_pickle=False)
                seq_idx += 1

        print(f"Done with chunk {chunk_idx} of {len(proc_seq_chunks)}")
        # print("Deleting...")
        # del model
        # del embedding
        # del input_ids
        # del attention_mask
        # del tokenizer
        # torch.cuda.empty_cache()
        # gc.collect()
        # time.sleep(5)
        chunk_idx += 1