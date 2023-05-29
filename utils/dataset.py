from datasets import load_dataset
from transformers import AutoTokenizer

def get_ptb(nsamples, seqlen, tokenizer, seed=0):

    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    nsamples = trainenc.input_ids.shape[1] if nsamples == -1 else nsamples

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    print('[ptb] Size of train loader: {} | test encoded test: {}'.format(len(trainloader), testenc.input_ids.shape[1]))
    return trainloader, testenc


if __name__ == '__main__':

    from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM

    model_name = 'facebook/opt-125m'
    n_samples = 10
    seqlen = 128

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    trainloader, testenc = get_ptb(nsamples=n_samples, seqlen=seqlen, tokenizer=tokenizer)
    # """
    for inputs, targets in trainloader:
        outputs = model(inputs)
        break
    # """