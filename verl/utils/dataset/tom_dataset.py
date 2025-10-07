import random

import torch
from torch.utils.data import Dataset
from datasets import dataset_dict, load_dataset
from pydantic.dataclasses import dataclass
from transformers import AutoTokenizer
import json

MAX_LEN = 512

SYSTEM_PROMPT = "Your are an expert linguist and communication assistant helping with research on the psychology of interactions. "
USER_TEMPLATE_LEAKAGE = """Below is a real conversation between two humans. Continue the conversation as realistically as possible. 
CONVERSATION:
{conv_hist}

Now respond with the following:
{responding_speaker}: {response}

{responding_speaker}: """

USER_TEMPLATE_BASELINE = """Below is a real conversation between two humans. 
Respond with the next utterance in the conversation as realistically as possible. 
CONVERSATION:
{conv_hist}

{responding_speaker}: """


USER_TEMPLATE = USER_TEMPLATE_LEAKAGE

NEGTOM_SYSTEM_PROMPT = "You are a helpful research assistant helping with human behavior reserach."

NEGTOM_USER_TEMPLATE = """{tom_prompt}
Based on the following conversation between humans, answer the below question.
{context}
{dialogue_history}
{question_details}"""

class NegTOMDataset(Dataset):

    def __init__(self, path: str, tokenizer: AutoTokenizer, config: dataclass = None):

        self.path = path
        self.tokenizer = tokenizer
        self.config = config or {}
        self.limit = self.config.get("limit", None)
        self.system_prompt = self.config.get("system_prompt", NEGTOM_SYSTEM_PROMPT)
        self.user_template = self.config.get("user_template", NEGTOM_USER_TEMPLATE)
        self.pp_baseline_prompt = "The assistant directly respond to the question and the response is enclosed within <answer> </answer> tags."
        self.tom_prompt = "The assistant first reason about each agent's mental process including what information they know, what information they assume the other knows, what their intents are, and what their strategy next is. Then provide the answer based on this reasoning. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
        self.context = "Background: Here is a negotiation conversation for a camping trip. There are two agents who own some basic supplies and negotiate with each other to split the additional food packages, water bottles, and firewood to make their camping trip even better. Each of these items will be of either High, Medium or Low priority for these two agents. Each of the additional items only has an available quantity of 3. "
        self.question_detail = "Now respond what {responding_speaker} will say in the next turn."
        self.generation_prefix = "<think>"
        self.limit_turn = self.config.get("include_turn", None)
        self.max_padding = 0
        self.seed = 143

        # TOM attributes
        self.data_source = "tomi"
        self.inc_hist = self.config.get("inc_hist", True)
        self.inc_ego_desire = self.config.get("inc_ego_desire", True)
        self.inc_ego_belief= self.config.get("inc_ego_belief", True)
        self.inc_other_belief= self.config.get("inc_other_belief", True)
        self.inc_ego_intent = self.config.get("inc_ego_intent", True)

        # Parse to prompts
        self.turns = self.load_data()
        self.prompts = [self.build_prompt(idx) for idx in range(len(self.turns))]

        print(f"Maximum length in dataset: {self.max_padding}")

    def load_data(self):

        with open(self.path, 'r') as inp:
            content = json.load(inp)

        conv = {}
        for turn in content:
            conv_id = turn['dialogue_id'].split("-")[0]
            conv[conv_id] = conv.get(conv_id, [])
            conv[conv_id].append(turn)

        turns = []
        for idx, c in conv.items():
            for first, second in zip(c, c[1:]):
                utter_idx = int(first['dialogue_id'].split("-")[1])
                if self.limit_turn is not None and utter_idx <  self.limit_turn: 
                    continue
                turns.append((first, second))
            
        if self.limit and len(turns) > self.limit:
            turns = turns[:self.limit]
                

        return turns
    
    # Not used. Was used to check oracle TOM affect of PP
    def get_tom_fill(self, idx):

       # Retrieve the turn at idx. It's stiched in two parts because we need information from both
        first, second = self.turns[idx]

        # Get features from the turn
        conv_id = int(first['dialogue_id'].split("-")[1])
        turn_idx = len(first['dialogue'])
        ego = second['utterance1_agent'] # -> agent1, agen2
        other =  second['utterance2_agent']
        if ego == "None":
            raise ValueError("Cannot have next turn agent empty")
        if other == "None":
            print("Warning, manually filling agent2.")
            other = "agent2"

        egokey = ego.replace("_", "")
        otherkey = other.replace("_", "")
        dialogue_hist = "\n".join(first['dialogue'])

        ego_desire = first[f'{egokey}_desire']

        # Build belief of ego
        ego_belief = "{"
        for s in ['high', 'medium', 'low']:
            ego_belief +=  s + ": " + first[f'{egokey}_belief_{s}'] + "\n"
        ego_belief += "}\n"

        # Build belief of other
        other_belief = "{"
        for s in ['high', 'medium', 'low']:
            other_belief +=  s + ": " + first[f'{otherkey}_belief_{s}'] + "\n"
        other_belief += "}\n"

        response = second['dialogue'][turn_idx] # the index is the the +1 of the last index

        ego_intent = second['utterance1_intent']

        prompt_detail = ""
        if self.inc_hist:
           prompt_detail += """
Dialogue History: 
{conv_hist}
""".format(conv_hist=dialogue_hist)

        if self.inc_ego_desire:
           prompt_detail += """

Desires for {responding_speaker}:
{ego_desire}""".format(responding_speaker=ego, ego_desire=ego_desire)

        if self.inc_ego_belief:
           prompt_detail += """
What {responding_speaker} thinks the desires of {other_speaker} are:
{ego_belief}""".format(responding_speaker=ego, other_speaker=other, ego_belief=ego_belief)

        if self.inc_other_belief:
           prompt_detail += """
What {other_speaker} thinks the desires of {responding_speaker} are:
{other_belief}""".format(responding_speaker=ego, other_speaker=other, other_belief=other_belief)

        if self.inc_ego_intent:
           prompt_detail += """
Based the above, {responding_speaker} will probably respond with the following goal:
{ego_intent}
""".format(responding_speaker=ego, ego_intent=ego_intent)
           
        return prompt_detail

    
    def build_prompt(self, idx):
        # Retrieve the turn at idx. It's stiched in two parts because we need information from both
        first, second = self.turns[idx]

        # Get features from the turn
        conv_id = int(first['dialogue_id'].split("-")[1])
        turn_idx = len(first['dialogue'])
        ego = second['utterance1_agent'] # -> agent1, agen2
        other =  second['utterance2_agent']
        if ego == "None":
            raise ValueError("Cannot have next turn agent empty")
        if other == "None":
            print("Warning, end of conversation without 2nd part. manually filling other part as 'agent2'.")
            other = "agent2"

        egokey = ego.replace("_", "")
        otherkey = other.replace("_", "")
        dialogue_hist = "\n".join(first['dialogue'])

        ego_desire = first[f'{egokey}_desire']

        # Build belief of ego
        ego_belief = "{"
        for s in ['high', 'medium', 'low']:
            ego_belief +=  s + ": " + first[f'{egokey}_belief_{s}'] + "\n"
        ego_belief += "}\n"

        # Build belief of other
        other_belief = "{"
        for s in ['high', 'medium', 'low']:
            other_belief +=  s + ": " + first[f'{otherkey}_belief_{s}'] + "\n"
        other_belief += "}\n"

        next_turn = second['dialogue'][turn_idx].split(": ") # the index is the the +1 of the last index
        if len(next_turn) != 2:
            print(f"Unexpect turn format (agent:text), got - {second['dialogue'][turn_idx]}")
        response = "".join(next_turn[1:])

        # We add tags for PP calculations since this is what we expect the model to produce
        response_with_tags = "<answer>" + response + "</answer>"
        response_word_len = len(response.split(" "))

        ego_intent = second['utterance1_intent']

        # Build the prompt
        dialogue_history = """Dialogue History:
        {dialogue_hist}
        """.format(dialogue_hist=dialogue_hist)

        context = ""
        if self.context != "":
            context =  self.context

        question_details = self.question_detail.format(responding_speaker=ego)

        # The final prompt
        conv_prompt = self.user_template.format(dialogue_history=dialogue_history, tom_prompt=self.pp_baseline_prompt, context=context, question_details=question_details)
        conv_prompt_tom = self.user_template.format(dialogue_history=dialogue_history, tom_prompt=self.tom_prompt, context=context, question_details=question_details)
        

        # Convert to huggingface messages to use for RL training
        messages_inp_tom_raw = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conv_prompt_tom},
        ]

        messages_inp_tom = self.tokenizer.apply_chat_template(
            messages_inp_tom_raw,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Add prefix if passed
        messages_inp_tom += self.generation_prefix

        # Convert to huggingface messages and tokenize for PP calculations
        messages_inp = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conv_prompt},
        ]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conv_prompt},
            {"role": "assistant", "content": response_with_tags},
        ]

        # First, get the conversation history tokens
        inp_tokens = self.tokenizer.apply_chat_template(
            messages_inp,
            return_tensors="pt",
            #padding='max_length',
            #max_length=2048,
            #truncate='max_length',
            add_generation_prompt=False,
        )

        input_len = inp_tokens.shape[-1]

        # then, get the all conversation tokens
        full_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            #padding='max_length',
            #max_length=2048,
            #truncate='max_length',
            add_generation_prompt=False,
            # return_dict=True,
            # return_assistant_tokens_mask=True,) #not working
        )

        full_tokens_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            return_tensors="pt",
            #padding='max_length',
            #max_length=2048,
            #truncate='max_length',
            add_generation_prompt=False,
            # return_dict=True,
            # return_assistant_tokens_mask=True,) #not working
        )

        full_len = full_tokens.shape[-1]

        # Update maximum length
        self.max_padding = max(self.max_padding, full_len)

        item = {'reward_model': {'ground_truth': response, 'style': 'rule'}, 'conv_id': conv_id, 'turn': turn_idx}

        return (inp_tokens, full_tokens, messages_inp_tom, full_tokens_str, input_len, full_len, response_word_len, self.system_prompt, conv_prompt_tom, item)
        

    def __getitem__(self, idx):
        
        # Get record
        inp_tokens, full_tokens, messages_inp_tom, full_tokens_str, input_len, full_len, response_word_len, system_prompt, conv_prompt_tom, item = self.prompts[idx]

        # Padding
        input_ids = torch.zeros([self.max_padding], dtype=torch.int64)
        input_ids[:] = self.tokenizer.pad_token_id
        input_ids[:full_len] = full_tokens.squeeze()

        response_mask = torch.zeros([self.max_padding])
        response_mask[input_len-1:full_len] = 1

        #response_mask = torch.ones_like(full_tokens)
        #response_mask[..., :input_len] = 0


        return {
            #"prompt_ids": inp_tokens.squeeze(),
            "input_ids": input_ids[:-1],#full_tokens.squeeze()[:-1],
            "labels": input_ids[1:],#full_tokens.squeeze()[1:],
            "response_mask": response_mask.squeeze()[1:],
            "prompt_len": input_len,
            "response_words": response_word_len,
            "data_source": self.data_source,
            "prompt": messages_inp_tom,
            "raw_system_prompt": system_prompt,
            "raw_user_prompt": conv_prompt_tom, 
            "prompt_for_pp": full_tokens_str,
            "ability": "theory_of_mind",
            "reward_model": item['reward_model'],
            "metadata": item,
        }

    def __len__(self):
        # Return the number of conversations in the dataset
        return len(self.prompts)


class TOMDialogue(Dataset):
    """
    A dataset for Theory of Mind (ToM) dialogue based on huggingface datasets.
    """

    def __init__(self, name: str, tokenizer: AutoTokenizer, config: dataclass = None):
        """
        Args:
            name (str): Path to dataset file.
            tokenizer (Tokenizer): Tokenizer for encoding text.
            config (dataclass): Configuration object.
        """

        self.name = name
        self.tokenizer = tokenizer
        self.config = config or {}
        self.limit = self.config.get("limit", None)
        self.system_prompt = self.config.get("system_prompt", SYSTEM_PROMPT)
        self.user_template = self.config.get("user_template", USER_TEMPLATE)
        self.split_at = 1 #self.config.get("split_at", None)
        self.max_padding = 0
        self.seed = 143
        
        random.seed(self.seed)

        # Load from huggingface
        self.dataset = load_dataset(name)

        self.convs = self.load_data(self.dataset["train"])
        self.prompts = [self._get_prompt(idx) for idx in range(len(self.convs))]

        print(f"Maximum length in dataset: {self.max_padding}")
        
    def load_data(self, dataset: dataset_dict.DatasetDict):
        # loads dataset and returns the formatted verison (Merge speaker ID to each turn)
        convs = []
        loaded = 0
        for d in dataset:
            turns = []
            for s, t in zip(d["speaker"], d["dialogue"]):
                turns.append((s, t))
            if len(turns) < 2:
                print("Skipping conversation of less than 2 turns")
                continue
            convs.append({"metadata": d, "turns": turns})
            if self.limit and loaded == self.limit:
                break

        return convs
    

    def _get_prompt(self, idx):
       # Get a single conversation
        conv = self.convs[idx]
        turns = conv["turns"]
        metadata = conv["metadata"]

        # Format example
        # Get random split:

        split_at = self.split_at or random.randint(0, len(turns) - 2)
        dialogue_hist = "\n".join(
            [turn[0] + ": " + turn[1] for turn in turns[: split_at + 1]]
        )
        speaker = turns[split_at + 1][0]
        response = turns[split_at + 1][1]
        response_word_len = len(response.split(" "))

        conv_prompt = self.user_template.format(
             conv_hist=dialogue_hist, responding_speaker=speaker, response=response
        )

        # response2 = c[split_at+2][0] + ": " + c[split_at+2][1]

        # Convert to huggingface messages
        messages_inp = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conv_prompt},
        ]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": conv_prompt},
            {"role": "assistant", "content": response},
        ]

        item = {
            "metadata": conv["metadata"],
            "dialogue": dialogue_hist,
            "true_response": response,
            "split_after": split_at,
            "total_turns": len(turns),
            "prompt": conv_prompt,
            "messages": messages,
        }

        # First, get the conversation history tokens
        inp_tokens = self.tokenizer.apply_chat_template(
            messages_inp,
            return_tensors="pt",
            #padding='max_length',
            #max_length=2048,
            #truncate='max_length',
            add_generation_prompt=False,
        )

        input_len = inp_tokens.shape[-1]

        # then, get the all conversation tokens
        full_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            #padding='max_length',
            #max_length=2048,
            #truncate='max_length',
            add_generation_prompt=False,
            # return_dict=True,
            # return_assistant_tokens_mask=True,) #not working
        )

        full_len = full_tokens.shape[-1]

        # Update maximum length
        self.max_padding = max(self.max_padding, full_len)

        return (inp_tokens, full_tokens, input_len, full_len, response_word_len, item)

    def __getitem__(self, idx):
        
        # Get record
        inp_tokens, full_tokens, input_len, full_len, response_word_len, item = self.prompts[idx]

        # Padding
        input_ids = torch.zeros([self.max_padding], dtype=torch.int64)
        input_ids[:] = self.tokenizer.pad_token_id
        input_ids[:full_len] = full_tokens.squeeze()

        response_mask = torch.zeros([self.max_padding])
        response_mask[input_len-1:full_len] = 1

        #response_mask = torch.ones_like(full_tokens)
        #response_mask[..., :input_len] = 0

        return {
            #"prompt_ids": inp_tokens.squeeze(),
            "input_ids": input_ids[:-1],#full_tokens.squeeze()[:-1],
            "labels": input_ids[1:],#full_tokens.squeeze()[1:],
            "response_mask": response_mask.squeeze()[1:],
            "data_source": self.data_source,
            "prompt_len": input_len,
            "response_words": response_word_len,
            "metadata": item,
        }

    def __len__(self):
        # Return the number of conversations in the dataset
        return len(self.prompts)
