# Comparison Tables

## Overall Performance

| Method                | Accuracy   | Correct/Total   | Perplexity   |
|:----------------------|:-----------|:----------------|:-------------|
| ICL_mistral_zero_shot | 7.00%      | 14/200          | N/A          |
| ICL_mistral_3_shot    | 29.50%     | 59/200          | N/A          |
| ICL_gemma_zero_shot   | 28.50%     | 57/200          | N/A          |
| ICL_gemma_3_shot      | 42.00%     | 84/200          | N/A          |
| Finetuned_mistral     | 55.00%     | 110/200         | 1.47         |
| Finetuned_gemma       | 71.50%     | 143/200         | 1.50         |

## Per-Task Performance

| Method                | Task                   | Accuracy   |   Count |
|:----------------------|:-----------------------|:-----------|--------:|
| ICL_mistral_zero_shot | Outcome Prediction     | 11.83%     |      93 |
| ICL_mistral_zero_shot | Manipulation Detection | 3.00%      |     100 |
| ICL_mistral_zero_shot | User Classification    | 0.00%      |       7 |
| ICL_mistral_3_shot    | Outcome Prediction     | 8.60%      |      93 |
| ICL_mistral_3_shot    | Manipulation Detection | 50.00%     |     100 |
| ICL_mistral_3_shot    | User Classification    | 14.29%     |       7 |
| ICL_gemma_zero_shot   | Outcome Prediction     | 40.86%     |      93 |
| ICL_gemma_zero_shot   | Manipulation Detection | 14.00%     |     100 |
| ICL_gemma_zero_shot   | User Classification    | 71.43%     |       7 |
| ICL_gemma_3_shot      | Outcome Prediction     | 46.24%     |      93 |
| ICL_gemma_3_shot      | Manipulation Detection | 36.00%     |     100 |
| ICL_gemma_3_shot      | User Classification    | 71.43%     |       7 |
| Finetuned_mistral     | Outcome Prediction     | 39.78%     |      93 |
| Finetuned_mistral     | Manipulation Detection | 66.00%     |     100 |
| Finetuned_mistral     | User Classification    | 100.00%    |       7 |
| Finetuned_gemma       | Outcome Prediction     | 49.46%     |      93 |
| Finetuned_gemma       | Manipulation Detection | 90.00%     |     100 |
| Finetuned_gemma       | User Classification    | 100.00%    |       7 |

