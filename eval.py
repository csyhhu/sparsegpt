"""
A evaluation pipeline
"""
import argparse

from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM

from utils.dataset import get_ptb
from utils.eval import simple_eval
from models.QuantizedOPT import QuantizedOPTForCausalLM

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-name', '-m', type=str, default='facebook/opt-125m',
    help='model to load.'
)
parser.add_argument(
    '--dataset-name', '-d', type=str, choices=['wikitext2', 'ptb', 'c4'], default='ptb',
    help='Where to extract calibration data from.'
)
parser.add_argument(
    '--n-samples', '-ns', type=int, default=-1,
    help=''
)
parser.add_argument(
    '--seq-length', '-seq', type=int, default=-1,
    help=''
)
parser.add_argument(
   '--is-test', '-test', action='store_true',
   help='Test mode.'
)
parser.add_argument(
   '--use-INT8', '-int8', action='store_true',
   help='Use default INT8'
)
parser.add_argument(
   '--use-default-INT8', '-dint8', action='store_true',
   help='Use default INT8'
)

def get_model(model_name, is_test=False, use_INT8=False, use_default_INT8=False, device='cpu', **kwargs):

    if is_test:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    else:
        if use_default_INT8:
            # free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
            max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                load_in_8bit=True,
                max_memory=max_memory
            )
        elif use_INT8:
            config = AutoConfig.from_pretrained(model_name)
            config.bitW = 8
            config.bitA = 32
            config.vector_index = [0, 1]
            model = QuantizedOPTForCausalLM(config)
            model.load_state_dict()
        else:
            model = OPTForCausalLM.from_pretrained(
                model_name,
                torch_dtype='auto'
            ).to(device)

    return model


if __name__ == '__main__':

    import torch
    args = parser.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---
    # Configuration
    # ---
    model_name = args.model_name
    nsamples = args.n_samples
    seqlen = args.seq_length
    is_test = args.is_test
    use_INT8 = args.use_INT8

    model = get_model(model_name, is_test, use_INT8, device=dev)
    seqlen = model.config.max_position_embeddings if seqlen == -1 else seqlen

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    trainloader, testenc = get_ptb(nsamples=nsamples, seqlen=seqlen, tokenizer=tokenizer)

    ppl = simple_eval(model, testenc, seqlen, device=dev)
    print('Final ppl: {}'.format(ppl))