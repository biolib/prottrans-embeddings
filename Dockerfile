FROM nvidia/cuda:11.1-runtime-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt update
RUN apt install python3 python3-pip -y

RUN pip3 install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -q SentencePiece transformers
RUN pip install numpy

COPY download_model_and_tokenizer.py download_model_and_tokenizer.py
COPY download_model_and_tokenizer_2.py download_model_and_tokenizer_2.py

RUN python3 download_model_and_tokenizer.py
RUN python3 download_model_and_tokenizer_2.py

RUN pip install Biopython

COPY generate_embeddings.py generate_embeddings.py
COPY sample.fasta sample.fasta