# Load the dataset
import datasets
import transformers as T
import evaluate
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import sys
import pandas as pd
from pydantic import BaseModel


from verl.utils.dataset.tom_dataset import  TOMDialogue, NegTOMDataset


def _is_list_like_excluding_str(x):
    if isinstance(x, str):
        return False
    try:
        iter(x)
    except TypeError:
        return False
    return True


def uncollate(batch, to_cpu=True):
    """
    Slightly modified from: https://gist.github.com/jvasilakes/ea4a4b402104c20154d856ceece74382
    Modified from
    https://lightning-flash.readthedocs.io/en/stable/_modules/flash/core/data/batch.html#default_uncollate
    to work with arbitarily nested batches.
    
    >>> batch = {'x': [[0., 1., 1.], [1., 1., 0.]], "metadata": {"example_id": [0, 1]}}
    >>> uncollate(batch)
    [{'x': [0., 1., 1.], "metadata": {"example_id": 0}},
     {'x': [1., 1., 0.], "metadata": {"example_id": 1}}]
    This function is used to uncollate a batch into samples.
    The following conditions are used:
    - if the ``batch`` is a ``dict``, the result will be a list of dicts
    - if the ``batch`` is list-like, the result is guaranteed to be a list
    Args:
        batch: The batch of outputs to be uncollated.
    Returns:
        The uncollated list of predictions.
    Raises:
        ValueError: If input ``dict`` values are not all list-like.
        ValueError: If input ``dict`` values are not all the same length.
        ValueError: If the input is not a ``dict`` or list-like.
    """
    if isinstance(batch, dict):
        if any(not _is_list_like_excluding_str(sub_batch)
               for sub_batch in batch.values()):
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to be list-like.")  # noqa
        uncollated_vals = [uncollate(val) for val in batch.values()]
        if len(set([len(v) for v in uncollated_vals])) > 1:
            raise ValueError("When uncollating a dict, all sub-batches (values) are expected to have the same length.")  # noqa
        elements = list(zip(*uncollated_vals))
        return [dict(zip(batch.keys(), element)) for element in elements]
    if isinstance(batch, (list, tuple, torch.Tensor)):
        if isinstance(batch, torch.Tensor) and to_cpu:
            return list(batch.cpu().numpy())
        return list(batch)
    raise ValueError(
        "The batch of outputs to be uncollated is expected to be a `dict` or list-like "  # noqa
        f"(e.g. `Tensor`, `list`, `tuple`, etc.), but got input of type: {type(batch)}"  # noqa
    )


def answer_perplexity(batch: dict):
    """
    Given a dictionary with all information, calculates the probability of the answer
    Args:
        batch['response_mask']: torch.IntTensor [bs, sequence_len]
        batch['input_ids']: torch.FloatTensor [bs, sequence_len, vocab_size]
    Returns:
        Float: the perplexity with shape [bs]
    """

    # Pluck the answer part
    prompt_len = batch["prompt_len"]
    labels = batch["labels"]#[..., prompt_len:]
    logits = batch["model_outputs"].logits#[:, prompt_len:,:]
    response_mask = batch["response_mask"]#[:, prompt_len:]

    # Get answer labels. [bs, sequence_len]
    #response_indx = torch.where(batch["response_mask"] == 1)
    #labels = batch["input_ids"][..., response_indx]

    #labels = torch.gather(batch["input_ids"], dim=-1, index=response_indx)
    #labels = batch["input_ids"][response_indx].view(batch["response_mask"].shape[0],-1)

    #print(batch["input_ids"].shape)
    #print(labels.shape)

    # Get logits for answer tokens
    #response_logits = batch["model_outputs"].logits[:,response_indx,:]
    #response_logits = batch["model_outputs"].logits.gather(dim=1, index=labels.unsqueeze(-1))
    #logits = batch["model_outputs"].logits

    #print(logits.shape)

    # Calculate perplexity
    num_words = batch["response_words"]
    log_prob = logprobs_from_logits_v2(logits, labels, response_mask)
    #print(log_prob.shape)
    #print(log_prob)
    #zero_logprob = log_prob == 0
    #print(f"Zero tokens logprob: {zero_logprob.sum(-1)}")
    #print(f"Prompt len: {prompt_len}")
    
    log_prob = log_prob.sum(dim=-1)
    log_prob = log_prob / response_mask.sum(-1) # A more correct way is to divide by words not tokens e.g. (/ num_words)
    #print(f"Words: {num_words}")
    #print(log_prob)
    return torch.exp(-1 * log_prob)


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels, response_mask):
    """
    A memory efficient implementation of logprobs_from_logits (numerically stable softmax)
    Source: https://github.com/volcengine/verl/utils/torch_functional.py

    Args:
        logits: torch.FloatTensor [bs, seq_len, vocab_size]
        labels: torch.Int64 [bs, seq_len]
    
    Outputs:
        torch.FloatTensor [bs, seq_len] of the logprob of the logits at indices in labels
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(
            logits, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = (
            logits_labels - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(
            logits, labels
        ):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels * response_mask

class NegTOMConfig(BaseModel):
    include_turn: int = 2
    inc_hist: bool = True
    inc_ego_desire: bool = True
    inc_ego_belief: bool = False
    inc_other_belief: bool = False
    inc_ego_intent: bool = False # Setting this to true is borrowing from the future
    limit: int = 800

class DataConfig(BaseModel):
    batch_size: int = 8
    limit: int = None
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "data/"



def run(config = None):

    if not config:
        config = DataConfig()

    bs = config.batch_size
    limit = config.limit

    # Get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device =  torch.device('mps')
    else:
        device = torch.device('cpu')

    print(device)

    model_name = config.model_name

    model = T.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = T.AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Move model to device
    model = model.to(device)

    conf = NegTOMConfig().dict()

    ds = NegTOMDataset("data/NegotiationToM.json", tokenizer, conf)
    #ds = TOMDialogue("nayohan/multi_session_chat", tokenizer, config=None)

    data = DataLoader(ds, batch_size=bs, shuffle=False)
    total_processed = min(len(ds), limit or np.inf)

    print(f"Data size: {len(ds)}")
    print(f"Will process: {total_processed}")

    out_file = model_name.replace("/","-")
    output_file = f"{config.output_dir}/NegotiationToM_{out_file}_limit{total_processed}"

    iter = 0
    total_pp = 0
    processed = []
    with torch.no_grad():
        for b in data:
            #return b
            # Check if we're at limit
            iter += 1
            if limit and iter > (limit/bs):
                iter -= 1 # revert to older value for correct estimates
                break

            # Move to device
            b['input_ids'] = b['input_ids'].to(device)
            b['labels'] = b['labels'].to(device)
            b['response_mask'] = b['response_mask'].to(device)
            b['response_words'] = b['response_words'].to(device)

            # Forward
            b['model_outputs'] = model(b['input_ids'])

            # Calculate perplexity
            b['answer_pp'] = answer_perplexity(b)
            #b['answer_pp'] = torch.zeros(b['input_ids'].shape[0])

            # Drop model outputs to free memory (TODO: check if we need to detach)
            del b['model_outputs']
            del b['input_ids']
            del b['labels']
            del b['response_mask']
            #del b['metadata']

            # Log metrics
            total_pp += b['answer_pp'].sum().cpu()

            # Detach and add results
            tmp_df = pd.DataFrame(uncollate(b))
            tmp_df['prompt'] = tmp_df['prompt'].map(lambda x: [{'content': x, 'role': 'user'}])
            tmp_df['raw_prompt'] = tmp_df.apply(lambda x: [{'content': x['raw_system_prompt'], 'role': 'system'}, {'content': x['raw_user_prompt'], 'role': 'user'}], axis=1)
            processed.extend(tmp_df.to_dict(orient="records"))
    
    processed_df = pd.DataFrame(processed)
    print(processed_df.shape)
    print(processed_df.sample(1))

    print("Saving dataframe")
    #processed_df.to_json(output_file + ".jsonl", orient='records', lines=True, default_handler=str)
    processed_df.to_parquet(output_file + ".parquet")
    print(f"Saved to {output_file}")

    print(f"Averae pp for {len(ds)} examples: {total_pp / total_processed}")


if __name__ == "__main__":
    run()
