"""Utils function for evaluation"""
import torch
import torch.nn as nn
from utils.misc import progress_bar

def simple_eval(model, testenc, seqlen, device):

    testenc_ids = testenc.input_ids
    _, sample_length = testenc_ids.shape
    n_samples = sample_length // seqlen
    nlls = []

    for i in range(n_samples):

        batch = testenc_ids[:, (i * seqlen):((i + 1) * seqlen)].to(device)
        outputs = model(batch)
        lm_logits = outputs.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc_ids[:, (i * seqlen):((i + 1) * seqlen)][:, 1:].to(device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)

        progress_bar(i, n_samples, "nnl: {}".format(neg_log_likelihood))

    ppl = torch.exp(torch.stack(nlls).sum() / sample_length)
    return ppl


def efficient_eval(model, testenc, seqlen, device):
    """
    Efficient evaluation perform model inference in a GPU memory saving scheme (algorithm perspective).
    It first perform embedding and caching the results.
    Then it releases the GPU memory and process layer-wisely (memory release after each layer)
    :param model:
    :param testenc:
    :param seqlen:
    :param device:
    :return:
    """

    testenc_ids = testenc.input_ids
    _, sample_length = testenc_ids.shape
    n_samples = sample_length // seqlen
    nlls = []

    #
    layers = model.model.decoder.layers
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(device)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    cache = {'idx': 0, 'feature' : None, 'attention_mask': None}



if __name__ == '__main__':

    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModel, OPTForCausalLM
    # from utils.dataset import get_ptb
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "facebook/opt-125m"
    seqlen = 128

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    # model = AutoModel.from_config(config)
    # model = OPTForCausalLM.from_config(config)
    model.to(device)

    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    ppl = simple_eval(model, testenc, seqlen, device=device)
    print(ppl)