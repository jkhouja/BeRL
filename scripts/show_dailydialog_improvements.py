#!/usr/bin/env python3
"""
Comparison script showing the improvements in convert_dailydialog.py
"""

print("=" * 80)
print("CONVERT_DAILYDIALOG.PY - Before vs After Comparison")
print("=" * 80)

print("\n📊 CONFIGURATION OPTIONS")
print("-" * 80)

before = [
    "min_turns",
    "max_turns",
    "min_response_words",
    "max_response_words",
    "sample_size",
    "seed",
    "split",
]

after = [
    # Original options
    "min_turns",
    "max_turns",
    "min_response_words",
    "max_response_words",
    "sample_size",
    "seed",
    "split",
    # NEW: Prompt configuration
    "system_prompt",
    "user_template",
    "prompt_style",
    "system_prompt_style",
    "generation_prefix",
    "include_system_in_prompt",
    # NEW: Response formatting
    "add_response_tags",
    "response_tag_open",
    "response_tag_close",
    # NEW: Metadata
    "ability",
    "data_source",
    # NEW: Turn filtering
    "limit_turn",
    # NEW: Config file
    "config (dict or file path)",
]

print(f"\nBefore: {len(before)} options")
for opt in before:
    print(f"  ✓ {opt}")

print(f"\nAfter: {len(after)} options (+{len(after) - len(before)} new)")
for opt in after:
    if opt not in before:
        print(f"  🆕 {opt}")
    else:
        print(f"  ✓ {opt}")

print("\n\n🎨 PROMPT STYLES")
print("-" * 80)

print("\nSystem Prompt Styles:")
print("  • default: Basic conversational assistant")
print("  • research: Expert linguist and communication assistant")

print("\nUser Template Styles:")
print("  • simple: Basic prediction prompt")
print("  • detailed: Elaborate conversation continuation")
print("  • baseline: Minimal baseline prompt")

print("\n\n🔧 NEW FEATURES")
print("-" * 80)

features = {
    "Config File Support": {
        "description": "Load all settings from YAML or JSON files",
        "example": "python convert_dailydialog.py --config my_config.yaml"
    },
    "Custom System Prompts": {
        "description": "Override with your own system prompt",
        "example": "python convert_dailydialog.py --system-prompt 'Custom prompt'"
    },
    "Custom User Templates": {
        "description": "Define your own prompt template with placeholders",
        "example": "python convert_dailydialog.py --user-template 'Template with {dialogue_history}'"
    },
    "Response Tags": {
        "description": "Wrap responses in custom tags (e.g., <answer>...</answer>)",
        "example": "python convert_dailydialog.py --add-response-tags"
    },
    "Generation Prefix": {
        "description": "Add prefix to generation (e.g., '<think>' for CoT)",
        "example": "python convert_dailydialog.py --generation-prefix '<think>'"
    },
    "Turn Filtering": {
        "description": "Only include turns >= specified number",
        "example": "python convert_dailydialog.py --limit-turn 2"
    },
    "Flexible Metadata": {
        "description": "Customize ability and data_source fields",
        "example": "python convert_dailydialog.py --ability custom_ability"
    },
}

for i, (feature, details) in enumerate(features.items(), 1):
    print(f"\n{i}. {feature}")
    print(f"   {details['description']}")
    print(f"   Example: {details['example']}")

print("\n\n📝 USAGE EXAMPLES")
print("-" * 80)

examples = [
    {
        "name": "Basic Usage (Same as before)",
        "command": "python scripts/convert_dailydialog.py --sample-size 1000"
    },
    {
        "name": "Research-Style Prompts",
        "command": "python scripts/convert_dailydialog.py --system-prompt-style research --prompt-style detailed"
    },
    {
        "name": "Chain-of-Thought Format",
        "command": "python scripts/convert_dailydialog.py --generation-prefix '<think>' --add-response-tags"
    },
    {
        "name": "Using Config File",
        "command": "python scripts/convert_dailydialog.py --config my_config.yaml"
    },
    {
        "name": "Later Turns Only",
        "command": "python scripts/convert_dailydialog.py --limit-turn 2 --min-turns 6"
    },
]

for i, example in enumerate(examples, 1):
    print(f"\n{i}. {example['name']}")
    print(f"   $ {example['command']}")

print("\n\n🔄 SIMILARITY TO tom_dataset.py")
print("-" * 80)

similarities = [
    "✓ Config dict support for programmatic usage",
    "✓ Custom system prompts and user templates",
    "✓ Multiple predefined prompt styles",
    "✓ Generation prefix support (e.g., '<think>')",
    "✓ Response tag wrapping (e.g., '<answer>...</answer>')",
    "✓ Turn-level filtering (limit_turn parameter)",
    "✓ Flexible metadata (ability, data_source)",
    "✓ Same output format structure",
    "➕ YAML/JSON config file support (new!)",
    "➕ Comprehensive CLI arguments (new!)",
]

print("\nShared Features:")
for similarity in similarities:
    print(f"  {similarity}")

print("\n\n💡 KEY IMPROVEMENTS")
print("-" * 80)

improvements = [
    "🎯 Full parity with tom_dataset.py's configurability",
    "📁 Config file support for reproducible experiments",
    "🔧 Extensive CLI options for quick experimentation",
    "📚 Comprehensive documentation and examples",
    "🎨 Multiple prompt style presets",
    "🏷️  Flexible response formatting with tags",
    "🔍 Better turn-level control",
    "📊 Enhanced output metadata",
]

for improvement in improvements:
    print(f"  {improvement}")

print("\n" + "=" * 80)
print("See scripts/DAILYDIALOG_README.md for complete documentation")
print("=" * 80)
