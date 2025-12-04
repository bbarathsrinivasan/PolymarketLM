# Comprehensive Method Comparison

## Overall Accuracy Comparison

| Method                 | Accuracy   | Correct/Total   | Perplexity   |
|:-----------------------|:-----------|:----------------|:-------------|
| ICL Mistral Zero-shot  | 7.00%      | 14/200          | N/A          |
| ICL Mistral Few-shot   | 29.50%     | 59/200          | N/A          |
| ICL Gemma Zero-shot    | 28.50%     | 57/200          | N/A          |
| ICL Gemma Few-shot     | 42.00%     | 84/200          | N/A          |
| Fine-tuned Mistral     | 55.00%     | 110/200         | 1.47         |
| Fine-tuned Gemma       | 71.50%     | 143/200         | 1.50         |
| Fine-tuned Mistral RAG | 100.00%    | 5/5             | 3.40         |
| Fine-tuned Gemma RAG   | 100.00%    | 5/5             | 3.73         |

## Per-Task Accuracy Comparison

| Method                 | Task                   | Accuracy   |   Count |
|:-----------------------|:-----------------------|:-----------|--------:|
| ICL Mistral Zero-shot  | Outcome Prediction     | 11.83%     |      93 |
| ICL Mistral Zero-shot  | Manipulation Detection | 3.00%      |     100 |
| ICL Mistral Zero-shot  | User Classification    | 0.00%      |       7 |
| ICL Mistral Few-shot   | Outcome Prediction     | 8.60%      |      93 |
| ICL Mistral Few-shot   | Manipulation Detection | 50.00%     |     100 |
| ICL Mistral Few-shot   | User Classification    | 14.29%     |       7 |
| ICL Gemma Zero-shot    | Outcome Prediction     | 40.86%     |      93 |
| ICL Gemma Zero-shot    | Manipulation Detection | 14.00%     |     100 |
| ICL Gemma Zero-shot    | User Classification    | 71.43%     |       7 |
| ICL Gemma Few-shot     | Outcome Prediction     | 46.24%     |      93 |
| ICL Gemma Few-shot     | Manipulation Detection | 36.00%     |     100 |
| ICL Gemma Few-shot     | User Classification    | 71.43%     |       7 |
| Fine-tuned Mistral     | Outcome Prediction     | 39.78%     |      93 |
| Fine-tuned Mistral     | Manipulation Detection | 66.00%     |     100 |
| Fine-tuned Mistral     | User Classification    | 100.00%    |       7 |
| Fine-tuned Gemma       | Outcome Prediction     | 49.46%     |      93 |
| Fine-tuned Gemma       | Manipulation Detection | 90.00%     |     100 |
| Fine-tuned Gemma       | User Classification    | 100.00%    |       7 |
| Fine-tuned Mistral RAG | Outcome Prediction     | 100.00%    |       5 |
| Fine-tuned Mistral RAG | Manipulation Detection | 0.00%      |       0 |
| Fine-tuned Mistral RAG | User Classification    | 0.00%      |       0 |
| Fine-tuned Gemma RAG   | Outcome Prediction     | 100.00%    |       5 |
| Fine-tuned Gemma RAG   | Manipulation Detection | 0.00%      |       0 |
| Fine-tuned Gemma RAG   | User Classification    | 0.00%      |       0 |

