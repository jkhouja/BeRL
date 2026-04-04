#!/usr/bin/env python3
"""
Script to download and convert DailyDialog dataset to NegotiationToM format.

This script downloads the roskoN/dailydialog dataset from HuggingFace and converts
it to match the format used by NegotiationToM parquet files for RL training.
"""

import argparse
import random
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import zipfile
import tempfile
import os
import yaml
import json


# Default system prompts
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant helping with conversation analysis and generation."

RESEARCH_SYSTEM_PROMPT = "You are an expert linguist and communication assistant helping with research on the psychology of interactions."

# Default user templates
SIMPLE_USER_TEMPLATE = """Below is a real conversation between two people.
Based on the conversation history, predict what {responding_speaker} will say next.

Dialogue History:
{dialogue_history}

Now respond with what {responding_speaker} will say next."""

DETAILED_USER_TEMPLATE = """Below is a real conversation between two people. Continue the conversation as realistically as possible.

CONVERSATION:
{dialogue_history}

Now respond with the following:
{responding_speaker}: """

BASELINE_USER_TEMPLATE = """Below is a real conversation between two people.
Respond with the next utterance in the conversation as realistically as possible.

CONVERSATION:
{dialogue_history}

{responding_speaker}: """

# Prompt style mapping
PROMPT_STYLES = {
    'simple': SIMPLE_USER_TEMPLATE,
    'detailed': DETAILED_USER_TEMPLATE,
    'baseline': BASELINE_USER_TEMPLATE,
}

SYSTEM_PROMPT_STYLES = {
    'default': DEFAULT_SYSTEM_PROMPT,
    'research': RESEARCH_SYSTEM_PROMPT,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}. Use .yaml, .yml, or .json")

    return config or {}


class DailyDialogConverter:
    """Converts DailyDialog dataset to NegotiationToM format."""

    def __init__(
        self,
        min_turns: int = 4,
        max_turns: Optional[int] = None,
        min_response_words: int = 5,
        max_response_words: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = 42,
        split: str = "train",
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        prompt_style: str = "simple",
        system_prompt_style: str = "default",
        ability: str = "conversation_generation",
        data_source: str = "dailydialog",
        generation_prefix: str = "",
        include_system_in_prompt: bool = False,
        add_response_tags: bool = False,
        response_tag_open: str = "<answer>",
        response_tag_close: str = "</answer>",
        limit_turn: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the converter.

        Args:
            min_turns: Minimum number of turns in a conversation to include
            max_turns: Maximum number of turns (None for no limit)
            min_response_words: Minimum words in response to include
            max_response_words: Maximum words in response (None for no limit)
            sample_size: Number of examples to sample (None for all)
            seed: Random seed for reproducibility
            split: Dataset split to use ('train', 'validation', 'test')
            system_prompt: Custom system prompt (overrides system_prompt_style)
            user_template: Custom user template (overrides prompt_style)
            prompt_style: Prompt style to use ('simple', 'detailed', 'baseline')
            system_prompt_style: System prompt style ('default', 'research')
            ability: Ability tag for the dataset
            data_source: Data source identifier
            generation_prefix: Prefix to add to generation prompts
            include_system_in_prompt: Include system prompt in the prompt field
            add_response_tags: Wrap responses in tags (e.g., <answer>...</answer>)
            response_tag_open: Opening tag for responses
            response_tag_close: Closing tag for responses
            limit_turn: Only include turns >= this turn number
            config: Configuration dictionary (overrides all other args)
        """
        # Use config dict if provided
        if config:
            self.min_turns = config.get("min_turns", min_turns)
            self.max_turns = config.get("max_turns", max_turns)
            self.min_response_words = config.get("min_response_words", min_response_words)
            self.max_response_words = config.get("max_response_words", max_response_words)
            self.sample_size = config.get("sample_size", sample_size)
            self.seed = config.get("seed", seed)
            self.split = config.get("split", split)
            self.prompt_style = config.get("prompt_style", prompt_style)
            self.system_prompt_style = config.get("system_prompt_style", system_prompt_style)
            self.ability = config.get("ability", ability)
            self.data_source = config.get("data_source", data_source)
            self.generation_prefix = config.get("generation_prefix", generation_prefix)
            self.include_system_in_prompt = config.get("include_system_in_prompt", include_system_in_prompt)
            self.add_response_tags = config.get("add_response_tags", add_response_tags)
            self.response_tag_open = config.get("response_tag_open", response_tag_open)
            self.response_tag_close = config.get("response_tag_close", response_tag_close)
            self.limit_turn = config.get("limit_turn", limit_turn)

            # User template and system prompt
            self.user_template = config.get("user_template", user_template)
            self.system_prompt = config.get("system_prompt", system_prompt)
        else:
            self.min_turns = min_turns
            self.max_turns = max_turns
            self.min_response_words = min_response_words
            self.max_response_words = max_response_words
            self.sample_size = sample_size
            self.seed = seed
            self.split = split
            self.prompt_style = prompt_style
            self.system_prompt_style = system_prompt_style
            self.ability = ability
            self.data_source = data_source
            self.generation_prefix = generation_prefix
            self.include_system_in_prompt = include_system_in_prompt
            self.add_response_tags = add_response_tags
            self.response_tag_open = response_tag_open
            self.response_tag_close = response_tag_close
            self.limit_turn = limit_turn
            self.user_template = user_template
            self.system_prompt = system_prompt

        # Set defaults based on styles if not provided
        if self.user_template is None:
            self.user_template = PROMPT_STYLES.get(self.prompt_style, SIMPLE_USER_TEMPLATE)

        if self.system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT_STYLES.get(self.system_prompt_style, DEFAULT_SYSTEM_PROMPT)

        random.seed(self.seed)

    def download_dataset(self):
        """Download the DailyDialog dataset from HuggingFace."""
        print(f"Downloading roskoN/dailydialog dataset (split: {self.split})...")

        # Download the zip file for the specified split
        zip_filename = f"{self.split}.zip"
        zip_path = hf_hub_download(
            repo_id="roskoN/dailydialog",
            filename=zip_filename,
            repo_type="dataset"
        )
        print(f"Downloaded {zip_filename}")

        # Extract and parse the dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Read the dialog file
            dialog_file = os.path.join(temp_dir, self.split, "dialogues_" + self.split + ".txt")

            conversations = []
            with open(dialog_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Each line is a conversation with utterances separated by " __eou__ "
                        utterances = line.split(" __eou__ ")
                        # Remove empty utterances
                        utterances = [u.strip() for u in utterances if u.strip()]
                        if utterances:
                            conversations.append({'dialog': utterances})

        print(f"Loaded {len(conversations)} conversations")
        return conversations

    def filter_conversation(self, dialogue: List[str]) -> bool:
        """
        Check if a conversation meets the filtering criteria.

        Args:
            dialogue: List of utterances in the conversation

        Returns:
            True if conversation should be included, False otherwise
        """
        # Check minimum turns
        if len(dialogue) < self.min_turns:
            return False

        # Check maximum turns
        if self.max_turns and len(dialogue) > self.max_turns:
            return False

        # Check if all utterances meet word count criteria
        for utterance in dialogue:
            word_count = len(utterance.split())
            if word_count < self.min_response_words:
                return False
            if self.max_response_words and word_count > self.max_response_words:
                return False

        return True

    def create_training_examples(
        self,
        dialogue: List[str],
        conv_id: int
    ) -> List[Dict[str, Any]]:
        """
        Create training examples from a conversation.

        For each conversation, we create multiple training examples by:
        - Taking the dialogue history up to turn i
        - Using turn i+1 as the response to predict

        Args:
            dialogue: List of utterances in the conversation
            conv_id: Unique conversation ID

        Returns:
            List of training examples in NegotiationToM format
        """
        examples = []

        # Create examples for each turn (starting from turn 2)
        # We need at least 1 turn of history to predict the next turn
        for split_idx in range(1, len(dialogue)):
            # Skip turns before limit_turn if specified
            if self.limit_turn is not None and split_idx < self.limit_turn:
                continue

            # Build dialogue history
            dialogue_hist = []
            for i in range(split_idx):
                speaker = f"Speaker_{i % 2 + 1}"  # Alternate between Speaker_1 and Speaker_2
                dialogue_hist.append(f"{speaker}: {dialogue[i]}")

            # Get response
            response = dialogue[split_idx]
            response_words = len(response.split())
            responding_speaker = f"Speaker_{split_idx % 2 + 1}"

            # Build dialogue history string
            dialogue_history_str = "\n".join(dialogue_hist)

            # Build user prompt using template
            user_prompt = self.user_template.format(
                dialogue_history=dialogue_history_str,
                responding_speaker=responding_speaker,
            )

            # Create prompt in chat format
            prompt = [
                {'content': user_prompt, 'role': 'user'}
            ]

            # Add system prompt if configured
            if self.include_system_in_prompt:
                prompt.insert(0, {'content': self.system_prompt, 'role': 'system'})

            # Create raw prompt (always includes system prompt)
            raw_prompt = [
                {'content': self.system_prompt, 'role': 'system'},
                {'content': user_prompt, 'role': 'user'}
            ]

            # Add response tags if configured
            response_with_tags = response
            if self.add_response_tags:
                response_with_tags = f"{self.response_tag_open}{response}{self.response_tag_close}"

            # Create example in NegotiationToM format
            example = {
                'prompt_len': None,  # Will be calculated during tokenization
                'response_words': response_words,
                'data_source': self.data_source,
                'prompt': prompt,
                'raw_system_prompt': self.system_prompt,
                'raw_user_prompt': user_prompt,
                'prompt_for_pp': None,  # Will be filled during tokenization
                'ability': self.ability,
                'reward_model': {
                    'ground_truth': response,
                    'style': 'rule'
                },
                'metadata': {
                    'conv_id': conv_id,
                    'turn': split_idx,
                    'total_turns': len(dialogue),
                    'responding_speaker': responding_speaker,
                    'dialogue_history': dialogue_history_str,
                },
                'answer_pp': None,  # Will be calculated later if needed
                'raw_prompt': raw_prompt,
            }

            # Add generation prefix if specified
            if self.generation_prefix:
                example['generation_prefix'] = self.generation_prefix

            # Add response with tags if configured
            if self.add_response_tags:
                example['response_with_tags'] = response_with_tags

            examples.append(example)

        return examples

    def convert(self, conversations) -> pd.DataFrame:
        """
        Convert the DailyDialog dataset to NegotiationToM format.

        Args:
            conversations: List of conversation dicts with 'dialog' key

        Returns:
            DataFrame in NegotiationToM format
        """
        all_examples = []
        conv_id = 0
        filtered_count = 0

        print("Converting conversations to training examples...")

        for item in tqdm(conversations):
            dialogue = item['dialog']

            # Filter conversation
            if not self.filter_conversation(dialogue):
                filtered_count += 1
                continue

            # Create training examples
            examples = self.create_training_examples(dialogue, conv_id)
            all_examples.extend(examples)
            conv_id += 1

        print(f"Filtered out {filtered_count} conversations")
        print(f"Created {len(all_examples)} training examples from {conv_id} conversations")

        # Sample if requested
        if self.sample_size and len(all_examples) > self.sample_size:
            print(f"Sampling {self.sample_size} examples from {len(all_examples)}")
            all_examples = random.sample(all_examples, self.sample_size)

        # Convert to DataFrame
        df = pd.DataFrame(all_examples)
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert DailyDialog dataset to NegotiationToM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python convert_dailydialog.py --output-name DailyDialog_train_limit1000 --sample-size 1000

  # Use config file
  python convert_dailydialog.py --config dailydialog_config_example.yaml

  # Use research-style prompts
  python convert_dailydialog.py --system-prompt-style research --prompt-style detailed

  # Add response tags for training
  python convert_dailydialog.py --add-response-tags --response-tag-open "<answer>" --response-tag-close "</answer>"

  # Custom prompts
  python convert_dailydialog.py --system-prompt "Custom system prompt" --user-template "Custom {dialogue_history} template"

  # Limit to specific turns
  python convert_dailydialog.py --limit-turn 2 --max-turns 10
        """
    )

    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON). Overrides all other arguments."
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for the converted dataset"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="DailyDialog",
        help="Base name for output file (will be saved as {output_name}.parquet)"
    )

    # Filtering options
    parser.add_argument(
        "--min-turns",
        type=int,
        default=4,
        help="Minimum number of turns in a conversation (default: 4)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns in a conversation (default: None)"
    )
    parser.add_argument(
        "--min-response-words",
        type=int,
        default=5,
        help="Minimum number of words in a response (default: 5)"
    )
    parser.add_argument(
        "--max-response-words",
        type=int,
        default=None,
        help="Maximum number of words in a response (default: None)"
    )
    parser.add_argument(
        "--limit-turn",
        type=int,
        default=None,
        help="Only include turns >= this turn number (default: None)"
    )

    # Sampling options
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of examples to sample (default: None, use all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: train)"
    )

    # Prompt configuration
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (overrides --system-prompt-style)"
    )
    parser.add_argument(
        "--system-prompt-style",
        type=str,
        default="default",
        choices=list(SYSTEM_PROMPT_STYLES.keys()),
        help="System prompt style (default: default)"
    )
    parser.add_argument(
        "--user-template",
        type=str,
        default=None,
        help="Custom user template (overrides --prompt-style). Use {dialogue_history} and {responding_speaker} placeholders"
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="simple",
        choices=list(PROMPT_STYLES.keys()),
        help="Prompt style (default: simple)"
    )
    parser.add_argument(
        "--generation-prefix",
        type=str,
        default="",
        help="Prefix to add to generation prompts (default: empty)"
    )
    parser.add_argument(
        "--include-system-in-prompt",
        action="store_true",
        help="Include system prompt in the prompt field (default: False)"
    )

    # Response formatting
    parser.add_argument(
        "--add-response-tags",
        action="store_true",
        help="Wrap responses in tags (default: False)"
    )
    parser.add_argument(
        "--response-tag-open",
        type=str,
        default="<answer>",
        help="Opening tag for responses (default: <answer>)"
    )
    parser.add_argument(
        "--response-tag-close",
        type=str,
        default="</answer>",
        help="Closing tag for responses (default: </answer>)"
    )

    # Dataset metadata
    parser.add_argument(
        "--ability",
        type=str,
        default="conversation_generation",
        help="Ability tag for the dataset (default: conversation_generation)"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="dailydialog",
        help="Data source identifier (default: dailydialog)"
    )

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
        # Extract output settings from config
        output_dir = config.pop('output_dir', args.output_dir)
        output_name = config.pop('output_name', args.output_name)
    else:
        config = None
        output_dir = args.output_dir
        output_name = args.output_name

    # Create converter (config overrides command line args)
    if config:
        converter = DailyDialogConverter(config=config)
    else:
        converter = DailyDialogConverter(
            min_turns=args.min_turns,
            max_turns=args.max_turns,
            min_response_words=args.min_response_words,
            max_response_words=args.max_response_words,
            sample_size=args.sample_size,
            seed=args.seed,
            split=args.split,
            system_prompt=args.system_prompt,
            user_template=args.user_template,
            prompt_style=args.prompt_style,
            system_prompt_style=args.system_prompt_style,
            ability=args.ability,
            data_source=args.data_source,
            generation_prefix=args.generation_prefix,
            include_system_in_prompt=args.include_system_in_prompt,
            add_response_tags=args.add_response_tags,
            response_tag_open=args.response_tag_open,
            response_tag_close=args.response_tag_close,
            limit_turn=args.limit_turn,
        )

    # Download dataset
    dataset = converter.download_dataset()

    # Convert dataset
    df = converter.convert(dataset)

    # Save to parquet
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{output_name}.parquet"
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)

    print(f"✓ Successfully saved {len(df)} examples to {output_path}")
    print("\nDataset info:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
    print(f"  - Data source: {df['data_source'].unique()}")
    print(f"  - Ability: {df['ability'].unique()}")
    print(f"\nConfiguration:")
    print(f"  - System prompt style: {converter.system_prompt_style}")
    print(f"  - Prompt style: {converter.prompt_style}")
    print(f"  - Generation prefix: '{converter.generation_prefix}'")
    print(f"  - Response tags: {converter.add_response_tags}")
    if converter.add_response_tags:
        print(f"    - Open tag: {converter.response_tag_open}")
        print(f"    - Close tag: {converter.response_tag_close}")
    print(f"\nSample statistics:")
    print(f"  - Avg response words: {df['response_words'].mean():.1f}")
    print(f"  - Min response words: {df['response_words'].min()}")
    print(f"  - Max response words: {df['response_words'].max()}")
    print(f"  - Unique conversations: {df['metadata'].apply(lambda x: x['conv_id']).nunique()}")


if __name__ == "__main__":
    main()
