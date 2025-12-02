# Documentation Index

Welcome to the CosyVoice Emotional Fine-Tuning documentation!

**Last Updated**: 2025-12-01

---

## Quick Navigation

### 🚀 Getting Started
- [Quick Reference Guide](QUICK_REFERENCE.md) - Fast command lookups
- [System Configuration](SYSTEM_CONFIGURATION.md) - Hardware & software setup
- [Dataset Documentation](DATASET_DOCUMENTATION.md) - Dataset details

### 📚 Training Guides
- [Training Guide](TRAINING_GUIDE.md) - Detailed training instructions
- [Parallel Training Guide](PARALLEL_TRAINING_GUIDE.md) - Multi-GPU comprehensive guide
- [Inference Guide](INFERENCE_GUIDE.md) - Running inference and generation

### 🔧 Technical Guides
- [Project Overview](../CLAUDE.md) - Project instructions and guidelines
- [Build RTX 5090 Docker](BUILD_RTX5090_DOCKER.md) - RTX 5090 specific setup
- [PyTorch 2.10 Compatibility](PYTORCH_2.10_COMPATIBILITY_FIXES.md) - Compatibility fixes
- [RTX 5090 PyTorch Upgrade](RTX_5090_PYTORCH_UPGRADE.md) - PyTorch upgrade guide
- [Common Issues](COMMON_ISSUES.md) - Troubleshooting common problems

---

## Documentation Structure

```
docs/
├── README.md                          # This file - documentation index
│
├── Quick Reference
│   ├── QUICK_REFERENCE.md             # Command quick reference
│   └── COMMON_ISSUES.md               # Troubleshooting guide
│
├── System & Configuration
│   ├── SYSTEM_CONFIGURATION.md        # Complete system specs
│   ├── BUILD_RTX5090_DOCKER.md        # RTX 5090 Docker setup
│   ├── RTX_5090_PYTORCH_UPGRADE.md    # PyTorch upgrade guide
│   └── PYTORCH_2.10_COMPATIBILITY...  # Compatibility fixes
│
├── Training Guides
│   ├── TRAINING_GUIDE.md              # Main training guide
│   ├── PARALLEL_TRAINING_GUIDE.md     # Multi-GPU training
│   └── INFERENCE_GUIDE.md             # Inference guide
│
├── Dataset & Models
│   ├── DATASET_DOCUMENTATION.md       # Dataset details
│   └── MODEL_COMPARISON_GUIDE.md      # Model comparison
│
Root:
└── CLAUDE.md                          # Project overview (root)
```

---

## Documentation by Topic

### System Configuration

**File**: [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md)

**Contents**:
- Hardware configuration (RTX 5090 specs)
- Software stack (Docker, Python, dependencies)
- Docker configuration and volume mounts
- Model architecture specifications
- Training configuration and hyperparameters
- Directory structure
- Performance benchmarks

**Use when**:
- Setting up a new machine
- Troubleshooting environment issues
- Understanding system requirements
- Reproducing the setup

### Dataset Information

**File**: [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md)

**Contents**:
- Dataset overview (ESD, RAVDESS, CREMA-D)
- Dataset statistics and distribution
- Data formats (JSONL, Kaldi, Parquet)
- Emotion categories and characteristics
- Preprocessing pipeline details
- Quality control guidelines
- Citation information

**Use when**:
- Understanding the dataset structure
- Preparing new data
- Analyzing emotion distribution
- Troubleshooting data issues
- Writing papers (citations)

### Quick Reference

**File**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**Contents**:
- Common Docker commands
- Training commands
- Monitoring commands
- Configuration quick edits
- Troubleshooting one-liners
- Useful shortcuts and aliases

**Use when**:
- Need quick command lookup
- During active training (keep open!)
- Troubleshooting common issues
- Setting up monitoring

### Project Instructions

**File**: [../CLAUDE.md](../CLAUDE.md)

**Contents**:
- Project overview
- Architecture details
- Environment setup
- Running inference
- Training pipeline
- Testing and linting
- Deployment options

**Use when**:
- First time setup
- Understanding CosyVoice architecture
- Learning best practices
- Deploying models

### Training Guide

**File**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

**Contents**:
- Detailed step-by-step training
- Data preparation walkthrough
- Training execution
- Model evaluation
- Common issues and solutions

**Use when**:
- Running training for the first time
- Step-by-step guidance needed
- Troubleshooting training failures

### Parallel Training Guide

**File**: [PARALLEL_TRAINING_GUIDE.md](PARALLEL_TRAINING_GUIDE.md)

**Contents**:
- Multi-machine setup
- Data sharing strategies
- Independent component training (LLM/Flow/HiFiGAN)
- Model collection and combination
- Monitoring multiple machines

**Use when**:
- Have multiple GPUs/machines
- Want 3x speedup
- Training all components (LLM + Flow + HiFiGAN)

### Inference Guide

**File**: [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)

**Contents**:
- Running inference with trained models
- Zero-shot voice cloning
- Emotion-controlled generation
- Audio quality optimization

**Use when**:
- Generating audio with trained models
- Testing model outputs
- Production deployment

---

## Common Workflows

### First-Time Setup

1. Read [CLAUDE.md](../CLAUDE.md) - Understand the project
2. Review [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md) - Check requirements
3. Follow [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Get started
4. Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - During training

### Training on Single GPU

1. [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md) - Verify setup
2. [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md) - Understand data
3. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Follow step-by-step
4. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Monitor training

### Training on Multiple GPUs

1. [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md) - Setup each machine
2. [PARALLEL_TRAINING_GUIDE.md](PARALLEL_TRAINING_GUIDE.md) - Detailed guide
3. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Monitor all machines

### Running Inference

1. [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Inference walkthrough
2. [MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md) - Compare models
3. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick inference commands

### Troubleshooting

1. [COMMON_ISSUES.md](COMMON_ISSUES.md) - Check known issues
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Common commands
3. [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md) - Verify environment
4. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Review training steps

---

## Document Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                        CLAUDE.md                            │
│                  (Project Overview)                         │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐   ┌──────────────┐   ┌──────────┐
    │  QUICK   │   │   TRAINING   │   │ PARALLEL │
    │  START   │   │    GUIDE     │   │ TRAINING │
    └──────────┘   └──────────────┘   └──────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
              ┌────────────────────────┐
              │  docs/ (Detailed Ref)  │
              └────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  SYSTEM  │   │ DATASET  │   │  QUICK   │
    │  CONFIG  │   │   DOCS   │   │   REF    │
    └──────────┘   └──────────┘   └──────────┘
```

---

## Reading Order Recommendations

### For Beginners
1. CLAUDE.md (understand the project)
2. QUICK_START.md (get running quickly)
3. QUICK_REFERENCE.md (keep handy during training)
4. TRAINING_GUIDE.md (if issues arise)

### For Experienced Users
1. QUICK_PARALLEL_SETUP.md (if multi-GPU)
2. QUICK_REFERENCE.md (command lookup)
3. SYSTEM_CONFIGURATION.md (for reproducibility)

### For Researchers
1. CLAUDE.md (architecture overview)
2. SYSTEM_CONFIGURATION.md (complete specs)
3. DATASET_DOCUMENTATION.md (data details + citations)
4. TRAINING_GUIDE.md (methodology)

### For Deployment
1. CLAUDE.md (deployment options)
2. SYSTEM_CONFIGURATION.md (requirements)
3. TRAINING_GUIDE.md (model export)

---

## Search by Keyword

### Commands
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

### Configuration
→ [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md)
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#configuration-quick-edits)

### Dataset
→ [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md)

### Docker
→ [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md#docker-configuration)
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#docker-commands)

### Emotions
→ [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md#emotion-categories)

### GPU / Hardware
→ [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md#hardware-configuration)

### Hyperparameters
→ [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md#hyperparameters)
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#configuration-quick-edits)

### Inference
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#inference-testing)
→ [CLAUDE.md](../CLAUDE.md#running-inference)

### Monitoring
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#monitoring)

### Parallel Training
→ [PARALLEL_TRAINING_GUIDE.md](PARALLEL_TRAINING_GUIDE.md)

### Performance
→ [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md#performance-benchmarks)
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#performance-benchmarks)

### Preprocessing
→ [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md#preprocessing-pipeline)
→ [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### Training
→ [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#training-commands)

### Troubleshooting
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#troubleshooting-commands)
→ [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md#troubleshooting-references)

---

## Additional Resources

### External Links

**CosyVoice Repository**:
- GitHub: https://github.com/FunAudioLLM/CosyVoice
- Issues: https://github.com/FunAudioLLM/CosyVoice/issues

**Dataset Sources**:
- ESD: [Emotional Speech Dataset](https://github.com/HLTSingapore/Emotional-Speech-Data)
- RAVDESS: [Ryerson Audio-Visual Database](https://zenodo.org/record/1188976)
- CREMA-D: [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)

**Technologies**:
- PyTorch: https://pytorch.org/
- Docker: https://www.docker.com/
- TensorBoard: https://www.tensorflow.org/tensorboard

### Support

**Questions?**
1. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common commands
2. Review [SYSTEM_CONFIGURATION.md](SYSTEM_CONFIGURATION.md) for setup details
3. Search existing issues on GitHub
4. Open a new issue with details

---

## Contributing to Documentation

### Documentation Standards

**File Naming**:
- Use UPPERCASE for top-level docs (CLAUDE.md, TRAINING_GUIDE.md)
- Use UPPERCASE for docs/ files (SYSTEM_CONFIGURATION.md)
- Be descriptive (not just "guide.md")

**Formatting**:
- Use Markdown
- Include table of contents for long docs
- Use code blocks with language hints
- Add last updated date
- Include examples

**Structure**:
- Start with overview
- Use clear headings (##, ###)
- Add cross-references to related docs
- Include troubleshooting section
- End with change log (for living docs)

### Updating Documentation

**When to update**:
- System configuration changes
- New features added
- Training procedure modified
- Bugs fixed (add to troubleshooting)
- Performance improvements

**How to update**:
1. Edit relevant .md file
2. Update "Last Updated" date
3. Add to change log (if applicable)
4. Update cross-references
5. Test all commands/code examples

---

## Document Status

| Document | Status | Last Updated | Completeness |
|----------|--------|--------------|--------------|
| docs/README.md | ✅ Complete | 2025-12-01 | 100% |
| docs/SYSTEM_CONFIGURATION.md | ✅ Complete | 2025-12-01 | 100% |
| docs/DATASET_DOCUMENTATION.md | ✅ Complete | 2025-12-01 | 100% |
| docs/QUICK_REFERENCE.md | ✅ Complete | 2025-12-01 | 100% |
| docs/COMMON_ISSUES.md | ✅ Complete | 2025-12-01 | 100% |
| docs/TRAINING_GUIDE.md | ✅ Complete | 2025-12-01 | 100% |
| docs/PARALLEL_TRAINING_GUIDE.md | ✅ Complete | 2025-12-01 | 100% |
| docs/INFERENCE_GUIDE.md | ✅ Complete | 2025-12-01 | 100% |
| docs/MODEL_COMPARISON_GUIDE.md | ✅ Complete | 2025-12-01 | 100% |
| docs/BUILD_RTX5090_DOCKER.md | ✅ Complete | 2025-12-01 | 100% |
| docs/RTX_5090_PYTORCH_UPGRADE.md | ✅ Complete | 2025-12-01 | 100% |
| docs/PYTORCH_2.10_COMPATIBILITY_FIXES.md | ✅ Complete | 2025-12-01 | 100% |
| CLAUDE.md | ✅ Complete | 2025-12-01 | 100% |

---

## Feedback

Found an error? Have suggestions?
- Open an issue on GitHub
- Submit a pull request
- Contact the project maintainers

---

**Happy Training!** 🚀

For quick command lookups, keep [QUICK_REFERENCE.md](QUICK_REFERENCE.md) open during training.
