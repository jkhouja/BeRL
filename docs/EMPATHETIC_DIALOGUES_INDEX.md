# Empathetic Dialogues Dataset - File Index

## 📁 All Created Files

### Core Scripts (3 files)

1. **`scripts/convert_empathetic_dialogues.py`** (650+ lines)
   - Main conversion script
   - Downloads facebook/empathetic_dialogues from HuggingFace
   - Converts to NegotiationToM parquet format
   - Supports multiple prompt styles and configurations

2. **`scripts/test_empathetic_dialogues_conversion.py`** (250+ lines)
   - Comprehensive test suite
   - Tests: basic conversion, empathy prompts, tags, filtering, config files
   - Validates output format and correctness

3. **`scripts/compare_conversation_datasets.py`** (280+ lines)
   - Side-by-side comparison of DailyDialog vs Empathetic Dialogues
   - Shows statistics, metadata differences, sample examples
   - Emotion distribution analysis

### Configuration Files (2 files)

4. **`scripts/empathetic_dialogues_config_example.yaml`**
   - YAML configuration template
   - Empathy-focused default settings
   - Commented and documented

5. **`scripts/empathetic_dialogues_config_example.json`**
   - JSON configuration template
   - Same settings as YAML version
   - Machine-readable format

### Example Scripts (1 file)

6. **`scripts/run_empathetic_dialogues_examples.sh`**
   - Bash script with 5 example conversions
   - Demonstrates different use cases
   - Ready to run out of the box

### Documentation Files (3 files)

7. **`scripts/README_EMPATHETIC_DIALOGUES.md`** (400+ lines)
   - Complete documentation
   - Configuration reference
   - Examples and use cases
   - Troubleshooting guide

8. **`EMPATHETIC_DIALOGUES_SUMMARY.md`** (300+ lines)
   - Overview of the implementation
   - Key differences from DailyDialog
   - Integration guide
   - Citation information

9. **`EMPATHETIC_DIALOGUES_QUICKSTART.md`** (250+ lines)
   - Quick start guide (5 minutes)
   - Common use cases
   - Troubleshooting
   - Checklists

### Index Files (1 file)

10. **`EMPATHETIC_DIALOGUES_INDEX.md`** (this file)
    - Complete file listing
    - Quick navigation
    - File descriptions

---

## 📊 File Organization

```
BeRL/
├── scripts/
│   ├── convert_empathetic_dialogues.py              # ⭐ Main converter
│   ├── test_empathetic_dialogues_conversion.py      # 🧪 Test suite
│   ├── compare_conversation_datasets.py             # 📊 Comparison tool
│   ├── empathetic_dialogues_config_example.yaml     # ⚙️ YAML config
│   ├── empathetic_dialogues_config_example.json     # ⚙️ JSON config
│   ├── run_empathetic_dialogues_examples.sh         # 📝 Examples
│   └── README_EMPATHETIC_DIALOGUES.md               # 📖 Full docs
├── EMPATHETIC_DIALOGUES_SUMMARY.md                  # 📄 Overview
├── EMPATHETIC_DIALOGUES_QUICKSTART.md               # 🚀 Quick start
└── EMPATHETIC_DIALOGUES_INDEX.md                    # 📁 This file
```

---

## 🔍 Quick Reference

### What Do I Need?

| I Want To... | File to Use |
|-------------|-------------|
| Convert the dataset | `scripts/convert_empathetic_dialogues.py` |
| Test the conversion | `scripts/test_empathetic_dialogues_conversion.py` |
| Compare datasets | `scripts/compare_conversation_datasets.py` |
| Use a config file | `scripts/empathetic_dialogues_config_example.yaml` |
| See examples | `scripts/run_empathetic_dialogues_examples.sh` |
| Read full docs | `scripts/README_EMPATHETIC_DIALOGUES.md` |
| Quick start (5 min) | `EMPATHETIC_DIALOGUES_QUICKSTART.md` |
| Understand implementation | `EMPATHETIC_DIALOGUES_SUMMARY.md` |

---

## 🚀 Quick Commands

### Convert Dataset
```bash
python scripts/convert_empathetic_dialogues.py \
    --system-prompt-style empathy \
    --prompt-style empathy
```

### Run Tests
```bash
python scripts/test_empathetic_dialogues_conversion.py
```

### Run Examples
```bash
bash scripts/run_empathetic_dialogues_examples.sh
```

### Compare with DailyDialog
```bash
python scripts/compare_conversation_datasets.py
```

---

## 📖 Documentation Guide

1. **First Time User?** → Start with `EMPATHETIC_DIALOGUES_QUICKSTART.md`

2. **Need Full Reference?** → Read `scripts/README_EMPATHETIC_DIALOGUES.md`

3. **Want to Understand Implementation?** → See `EMPATHETIC_DIALOGUES_SUMMARY.md`

4. **Looking for a Specific File?** → You're in the right place! (this file)

---

## 🎯 Features Summary

### Converter Features
- ✅ Downloads from HuggingFace
- ✅ Multiple prompt styles (simple, detailed, baseline, empathy)
- ✅ Multiple system prompt styles (default, research, empathy)
- ✅ Emotion context integration
- ✅ Configurable filtering (turns, words)
- ✅ Response tagging support
- ✅ Config file support (YAML/JSON)
- ✅ Sampling and seeding
- ✅ All dataset splits (train/validation/test)

### Test Coverage
- ✅ Basic conversion
- ✅ Empathy prompts
- ✅ Response tags
- ✅ Filtering options
- ✅ Config file loading

### Documentation
- ✅ Quick start guide
- ✅ Full reference documentation
- ✅ Implementation summary
- ✅ Example scripts
- ✅ Troubleshooting guide
- ✅ Citation information

---

## 📊 Statistics

- **Total Files**: 10
- **Total Lines of Code**: ~1,800+
- **Total Lines of Docs**: ~1,000+
- **Test Coverage**: 5 test scenarios
- **Examples**: 5+ usage examples
- **Config Formats**: YAML, JSON
- **Dataset Splits**: train, validation, test

---

## 🔗 Related Files (Existing)

For comparison with the similar DailyDialog implementation:

- `scripts/convert_dailydialog.py` - Similar converter for DailyDialog
- `scripts/test_dailydialog_conversion.py` - DailyDialog tests
- `scripts/dailydialog_config_example.yaml` - DailyDialog config
- `scripts/DAILYDIALOG_README.md` - DailyDialog docs

---

## ✅ Implementation Checklist

- [x] Main converter script
- [x] Test suite
- [x] Config files (YAML & JSON)
- [x] Example runner script
- [x] Full documentation
- [x] Quick start guide
- [x] Summary document
- [x] Comparison tool
- [x] Index file (this)

---

## 📚 External Resources

- **Dataset**: [facebook/empathetic_dialogues on HuggingFace](https://huggingface.co/datasets/facebook/empathetic_dialogues)
- **Paper**: [Towards Empathetic Open-domain Conversation Models (ACL 2019)](https://arxiv.org/abs/1811.00207)
- **Original Repo**: [ParlAI](https://github.com/facebookresearch/ParlAI)

---

## 🎓 Citation

```bibtex
@inproceedings{rashkin2019towards,
  title={Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset},
  author={Rashkin, Hannah and Smith, Eric Michael and Li, Margaret and Boureau, Y-Lan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={5370--5381},
  year={2019}
}
```

---

**Last Updated**: 2024
**Status**: ✅ Complete and tested
**Maintainer**: BeRL project
