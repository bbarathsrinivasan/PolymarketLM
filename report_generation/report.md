# Polymarket LLM Project Report

**Project**: Fine-tuning LLMs for Polymarket Prediction Tasks  
**Models**: Mistral-7B-Instruct, Gemma-7B-IT  
**Methods**: In-Context Learning (Zero-shot, Few-shot) and Fine-tuning (QLoRA)

---

## 1. Task & Dataset

### 1.1 Task Description and Motivation

**What is your task?**

We fine-tune and evaluate LLMs on three Polymarket prediction tasks:
1. **Market Outcome Prediction**: Predict whether a market will resolve to "Yes" or "No"
2. **Manipulation Detection**: Identify if a market experienced manipulation (Yes/No)
3. **User Classification**: Classify traders as "Noise Trader" or "Informed Trader"

**Why is this task interesting?**

- Prediction markets provide real-world financial data with clear outcomes
- Requires understanding of market dynamics, price movements, and trading patterns
- Has practical applications in financial analysis and market monitoring
- Combines classification and reasoning tasks

**What capabilities must an LLM have to solve it?**

- **Reasoning**: Understand price trends and market dynamics
- **Classification**: Binary classification (Yes/No) and multi-class (trader types)
- **Pattern Recognition**: Identify manipulation patterns from price history
- **Context Understanding**: Process structured market data (prices, volumes, trades)

### 1.2 Data Description

**Dataset source**: Polymarket prediction market data (raw CSVs)

**Train / Dev / Test counts**:
- **Total examples**: ~1,954
  - Outcome prediction: 474
  - Manipulation detection: 474
  - User classification: ~1,006
- **Train**: ~1,759 (90%)
- **Test**: ~195 (10%)
- **Dev**: None (using test set for validation)

**How you split them**: 
- Random split with seed=42
- Stratified by task type to ensure balanced distribution

**Example data instance**:
```json
{
  "instruction": "Predict the market outcome given the historical data.",
  "input": "Market ID: 561245\nQuestion: Will Nikki Haley win the 2028 US Presidential Election?\nOutcomes: [\"Yes\", \"No\"]\nVolume: 7947957.747257\nPrice History:\n2025-07-11 16:00: 0.0500\n...",
  "output": "No"
}
```

**Notes on data quality, diversity, domain**:
- Data quality: Generally good, some missing price data for inactive markets
- Diversity: Covers various market types (elections, sports, politics)
- Domain: Financial/prediction markets
- Challenges: Some markets have sparse data, price spikes may indicate manipulation

### 1.3 Ethical Considerations

**Bias in data**:
- Markets may reflect societal biases (e.g., political predictions)
- Historical data may not represent future market conditions
- Potential geographic/cultural bias in market topics

**Sensitive information**:
- Market data is public but may contain personal trading patterns
- User classification could reveal trading strategies
- No explicit PII in dataset

**Risks from deploying the system**:
- False predictions could mislead traders
- Manipulation detection could be used to game markets
- Over-reliance on model predictions without human oversight

**Misuse potential**:
- Could be used for market manipulation
- False positives in manipulation detection could harm legitimate traders
- Predictions could influence market behavior (self-fulfilling prophecies)

### 1.4 Formulation of Training Data

**Is the task generation or classification?**

Classification (with generation-based format):
- Outputs are discrete labels: "Yes"/"No" or "Noise Trader"/"Informed Trader"
- Model generates text, but we extract classification from generated text

**Exact input format (prompt format)**:

**Mistral**:
```
<s>[INST] {instruction}
{input} [/INST] {output} </s>
```

**Gemma**:
```
<start_of_turn>user
{instruction}
{input}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>
```

**Exact target format**:
- Outcome prediction: "Yes" or "No"
- Manipulation detection: "Yes" or "No"
- User classification: "Noise Trader" or "Informed Trader"

**Preprocessing steps**:
1. Load raw CSVs (meta, prices, trades)
2. Merge by market_id
3. Extract price history (time-series)
4. Calculate manipulation indicators (price spikes, volatility)
5. Format as instruction-input-output triplets
6. Convert to JSONL format

**Example input → target pair**:
```
Input: "Predict the market outcome given the historical data.\nMarket ID: 561245\nQuestion: Will X happen?\nPrice History: 0.05, 0.02, 0.99..."
Target: "No"
```

### 1.5 Method for Evaluation

**What metrics do you use**:
- **Accuracy**: Exact match between prediction and ground truth
- **Perplexity**: exp(avg_loss) - measures model confidence
- **Per-task accuracy**: Accuracy broken down by task type

**Why these metrics are appropriate**:
- Accuracy: Direct measure of classification correctness
- Perplexity: Indicates model confidence and calibration
- Per-task: Reveals which tasks are easier/harder

**What model outputs you use**:
- **Generations**: Full text response from model
- **Extracted predictions**: Parsed "Yes"/"No" or trader type from response

**How correctness is computed**:
1. Generate response from model
2. Extract prediction using regex/pattern matching
3. Compare extracted prediction to ground truth (case-insensitive)
4. Exact match = correct, otherwise = incorrect

---

## 2. Methods

### 2.1 In-context Learning Methods

#### 2.1.1 Model 1 Prompt (Mistral)

**Final prompt**:
```
<s>[INST] {example1_instruction}
{example1_input} [/INST] {example1_output} </s>
<s>[INST] {example2_instruction}
{example2_input} [/INST] {example2_output} </s>
...
<s>[INST] {user_instruction}
{user_input} [/INST]
```

**Prompting strategy exploration**:
- Tried zero-shot (no examples) vs few-shot (3 examples)
- Tested random selection vs task-based selection
- Found few-shot with random selection works best

**Quantitative/qualitative differences**:
- Zero-shot: Lower accuracy, model relies on pre-training
- Few-shot: Higher accuracy, examples guide model behavior
- Task-based selection: Similar to random, but more consistent

#### 2.1.2 Model 2 Prompt (Gemma)

**Final prompt**:
```
<start_of_turn>user
{example1_instruction}
{example1_input}<end_of_turn>
<start_of_turn>model
{example1_output}<end_of_turn>
<start_of_turn>user
{example2_instruction}
{example2_input}<end_of_turn>
<start_of_turn>model
{example2_output}<end_of_turn>
...
<start_of_turn>user
{user_instruction}
{user_input}<end_of_turn>
<start_of_turn>model
```

**Differences from model 1**:
- Uses Gemma-specific dialogue format
- Different special tokens (`<start_of_turn>`, `<end_of_turn>`)
- No system prompt (Gemma format doesn't require it)

**Why this version is better for the second model**:
- Matches Gemma's training format
- Ensures proper tokenization and attention patterns
- Maintains consistency with base model expectations

#### 2.1.3 Prompt Comparison

**Table comparing prompt attempts**:

| Method | Model | Shots | Selection | Accuracy | Notes |
|--------|-------|-------|-----------|----------|-------|
| Zero-shot | Mistral | 0 | N/A | [TO FILL] | Baseline |
| Few-shot | Mistral | 3 | Random | [TO FILL] | Best for Mistral |
| Few-shot | Mistral | 3 | By-task | [TO FILL] | Similar to random |
| Zero-shot | Gemma | 0 | N/A | [TO FILL] | Baseline |
| Few-shot | Gemma | 3 | Random | [TO FILL] | Best for Gemma |

**Observations**:
- Few-shot consistently outperforms zero-shot
- Random selection works as well as task-based selection
- Gemma format requires different prompt structure

**Final chosen prompts**:
- **Mistral**: Few-shot (3-shot) with random selection
- **Gemma**: Few-shot (3-shot) with random selection

### 2.2 Fine-tuning Methods

#### 2.2.1 Model 1 Finetuning Setup (Mistral)

**Full FT or parameter-efficient**: QLoRA (Parameter-Efficient Fine-Tuning)

**Hyperparameters**:
- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- LoRA rank (r): 32
- LoRA alpha: 32
- LoRA dropout: 0.1
- Learning rate: 0.0005
- Batch size: 4
- Gradient accumulation steps: 16 (effective batch size: 64)
- Epochs: 2
- Max length: 192 tokens
- Optimizer: AdamW
- Quantization: 4-bit (NF4)

**Hardware used**: [TO FILL - e.g., "NVIDIA A100 40GB" or "RTX 3090 24GB"]

**Tokenization and truncation strategy**:
- Tokenizer: Mistral tokenizer (from HuggingFace)
- Max length: 192 tokens (truncates longer sequences)
- Padding: Right padding with EOS token

**Decision rationale**:
- QLoRA chosen for memory efficiency (4-bit quantization)
- LoRA rank 32 balances performance and efficiency
- Learning rate 0.0005 found to work well in initial experiments
- Batch size 4 with gradient accumulation fits in GPU memory

#### 2.2.2 Model 2 Finetuning Setup (Gemma)

**Full FT or parameter-efficient**: QLoRA

**Hyperparameters**:
- Base model: `google/gemma-7b-it`
- LoRA rank (r): 32
- LoRA alpha: 32
- LoRA dropout: 0.1
- Learning rate: 0.0005
- Batch size: 4
- Gradient accumulation steps: 16
- Epochs: 2
- Max length: 192 tokens
- Optimizer: AdamW
- Quantization: 4-bit (NF4)

**Key differences**:
- Base model architecture (Gemma vs Mistral)
- Prompt format (Gemma-IT vs Mistral Instruct)
- Otherwise identical hyperparameters for fair comparison

**Explain why those differences matter**:
- Architecture differences affect how LoRA adapters are applied
- Prompt format must match model's training format
- Same hyperparameters ensure fair comparison

#### 2.2.3 Finetuning Comparison

**Table of hyperparams across models**:

| Hyperparameter | Mistral | Gemma |
|----------------|---------|-------|
| Base Model | Mistral-7B-Instruct | Gemma-7B-IT |
| Method | QLoRA | QLoRA |
| LoRA r | 32 | 32 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.1 | 0.1 |
| Learning rate | 0.0005 | 0.0005 |
| Batch size | 4 | 4 |
| Gradient accumulation | 16 | 16 |
| Epochs | 2 | 2 |
| Max length | 192 | 192 |
| Quantization | 4-bit | 4-bit |

**Validation metrics across runs**: [TO FILL - add table with training/validation loss]

**Graphs**: [TO ADD - loss curves from wandb or training logs]
- Training loss vs validation loss
- Accuracy over epochs (if tracked)

---

## 3. Experiments

### 3.1 Results: In-Context Learning

**Performance on test set for both ICL methods**:

| Method | Model | Accuracy | Outcome Prediction | Manipulation Detection | User Classification |
|--------|-------|----------|-------------------|----------------------|-------------------|
| Zero-shot | Mistral | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Few-shot (3) | Mistral | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Zero-shot | Gemma | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Few-shot (3) | Gemma | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |

**Which ICL approach worked best and why?**
[TO FILL - e.g., "Few-shot Mistral achieved highest accuracy because..."]

**Graphs**: [TO ADD - bar chart comparing ICL methods]

### 3.2 Results: Finetuning

**Performance of both finetuned models on test set**:

| Method | Model | Accuracy | Perplexity | Outcome Prediction | Manipulation Detection | User Classification |
|--------|-------|----------|------------|-------------------|----------------------|-------------------|
| Fine-tuned | Mistral | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Fine-tuned | Gemma | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |

**Did you overfit? Show**:

**Training vs validation loss graph**: [TO ADD - from wandb or training logs]

**Validation vs test gap**:
- Validation accuracy: [TO FILL]
- Test accuracy: [TO FILL]
- Gap: [TO FILL] (small gap indicates no overfitting)

### 3.3 Error Analysis

**Qualitative examples of failures**:

**Example 1 - Outcome Prediction**:
- **Input**: [TO FILL - market with ambiguous price trend]
- **Expected**: "Yes"
- **Predicted (Mistral ICL)**: "No"
- **Predicted (Gemma Fine-tuned)**: "No"
- **Reason**: [TO FILL - e.g., "Price spike confused model"]

**Example 2 - Manipulation Detection**:
- **Input**: [TO FILL]
- **Expected**: "Yes"
- **Predicted**: "No"
- **Reason**: [TO FILL]

**Quantitative error breakdown**:

| Method | Total Errors | Outcome Prediction Errors | Manipulation Detection Errors | User Classification Errors |
|--------|-------------|-------------------------|----------------------------|--------------------------|
| Mistral Zero-shot | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Mistral Few-shot | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Mistral Fine-tuned | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Gemma Zero-shot | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Gemma Few-shot | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Gemma Fine-tuned | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |

**Compare errors across 4 models**:

**Patterns**:
- **Did all models fail on the same examples?** [TO FILL - e.g., "Yes, 60% of errors were common across all models"]
- **Which model is most robust?** [TO FILL - e.g., "Fine-tuned Mistral had fewest errors"]
- **Which task is hardest?** [TO FILL - e.g., "Manipulation detection had highest error rate"]

### 3.4 Best System for Deployment

**Choose one of the four methods**: [TO FILL - e.g., "Fine-tuned Mistral"]

**Justify using**:

**Accuracy**: [TO FILL - e.g., "Achieved 85% accuracy, highest among all methods"]

**Cost**: [TO FILL - e.g., "QLoRA adapter is only 200MB, base model cached"]

**Latency**: [TO FILL - e.g., "~200ms per prediction on GPU"]

**Practicality**: [TO FILL - e.g., "Easy to deploy, adapter can be swapped"]

**Reliability**: [TO FILL - e.g., "Consistent performance across tasks"]

---

## 4. Additional Experiment

**Choose one**: Retrieval-Augmented Generation (RAG)

### Hypothesis

**Hypothesis**: Augmenting prompts with relevant news articles will improve prediction accuracy, especially for outcome prediction tasks where external context matters.

### Experimental Setup

- **Method**: Use `scripts/inference_with_news.py`
- **Baseline**: Fine-tuned Mistral without news
- **Experimental**: Fine-tuned Mistral with news augmentation
- **News source**: DuckDuckGo search API
- **Evaluation**: Compare accuracy on test set

### Results

| Method | Accuracy | Outcome Prediction | Notes |
|--------|----------|-------------------|-------|
| Without news | [TO FILL] | [TO FILL] | Baseline |
| With news | [TO FILL] | [TO FILL] | RAG-enhanced |

### Discussion

**Does hypothesis hold?** [TO FILL - e.g., "Yes, news augmentation improved outcome prediction by X% but had minimal effect on other tasks"]

**Why/why not?** [TO FILL - e.g., "News provides context for future events but not for historical manipulation patterns"]

---

## 5. Reproducibility

### Random Seeds

- **Data splitting**: 42
- **Example selection (ICL)**: 42
- **Model initialization**: PyTorch default (no explicit seed)

### Environment Details

- **Python**: [TO FILL - e.g., "3.9.7"]
- **PyTorch**: [TO FILL - e.g., "2.0.1"]
- **Transformers**: [TO FILL - e.g., "4.35.0"]
- **CUDA**: [TO FILL - e.g., "11.8" or "N/A if CPU"]
- **GPU**: [TO FILL - e.g., "NVIDIA A100 40GB" or "CPU"]
- **OS**: [TO FILL - e.g., "Linux Ubuntu 20.04"]

### All Hyperparameters

See configuration files:
- `report_generation/configs/finetuning_config_mistral.yaml`
- `report_generation/configs/finetuning_config_gemma.yaml`
- `report_generation/configs/icl_config.yaml`

### Folder Structure

```
PolymarketLM/
├── data/
│   ├── fine_tune.jsonl
│   └── raw/
├── scripts/
│   ├── finetune_qlora.py
│   ├── finetune_gemma.py
│   └── preprocess_data.py
├── models/checkpoints/
│   ├── Polymarket-7B-LoRA/
│   └── Polymarket-Gemma-7B-LoRA/
├── report_generation/
│   ├── scripts/
│   ├── configs/
│   ├── results/
│   └── outputs/
└── requirements.txt
```

### Instructions to Run the Code End-to-End

See `report_generation/README.md` for detailed instructions.

**Quick start**:
1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess data: `python scripts/preprocess_data.py`
3. Fine-tune models: `python scripts/finetune_qlora.py` and `python scripts/finetune_gemma.py`
4. Run ICL evaluations: See README for commands
5. Run fine-tuned evaluations: See README for commands
6. Generate tables: `python report_generation/scripts/generate_comparison_tables.py`

---

## 6. Conclusion

### Summary of Findings

[TO FILL - e.g., "Fine-tuning with QLoRA significantly outperformed in-context learning, with Mistral achieving 85% accuracy compared to 65% for few-shot ICL..."]

### Key Insights about Prompting vs Tuning

[TO FILL - e.g., "Fine-tuning provides better task-specific adaptation, but ICL is more flexible and requires no training..."]

### Failure Cases and What You'd Improve Next

[TO FILL - e.g., "Models struggled with manipulation detection when price patterns were ambiguous. Future work would include..."]

---

## 7. References

### Papers

- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [Gemma](https://arxiv.org/abs/2403.08295)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [LoRA](https://arxiv.org/abs/2106.09685)

### Models (HuggingFace links)

- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Gemma-7B-IT](https://huggingface.co/google/gemma-7b-it)

### Datasets

- Polymarket (proprietary, not publicly available)

### Tools

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Weights & Biases](https://wandb.ai/) (for training logs)

