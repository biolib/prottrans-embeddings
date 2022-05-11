import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
import gc

#tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

gc.collect()
