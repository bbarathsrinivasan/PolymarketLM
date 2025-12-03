# Memory Optimization Tips for Fine-tuning

## CUDA Out of Memory (OOM) Solutions

If you encounter OOM errors during training, try these solutions in order:

### 1. Reduce Batch Size and Increase Gradient Accumulation

**Current Gemma config** (already optimized):
- `batch_size: 2` (reduced from 4)
- `gradient_accumulation_steps: 8` (increased from 4)
- Effective batch size: 16 (same as before)

**To reduce further**:
```yaml
batch_size: 1
gradient_accumulation_steps: 16
```

### 2. Reduce Max Length

**Current**: `max_length: 256` (reduced from 512)

**To reduce further**:
```yaml
max_length: 192  # or even 128
```

### 3. Enable Gradient Checkpointing

Already enabled in scripts, but verify:
- Model: `model.gradient_checkpointing_enable()`
- TrainingArgs: `gradient_checkpointing=True`

### 4. Clear GPU Cache

The scripts now clear GPU cache before training. If you still have issues:

```python
import torch
torch.cuda.empty_cache()
```

### 5. Use 4-bit Quantization

Already enabled by default (`no_4bit: false`). Don't disable unless you have >24GB VRAM.

### 6. Reduce LoRA Rank

**Current**: `lora_r: 32`

**To reduce**:
```yaml
lora_r: 16  # or even 8
lora_alpha: 16  # Should match lora_r
```

### 7. Limit Dataset Size (for testing)

```yaml
max_samples: 1000  # Test with smaller dataset first
```

### 8. Set Environment Variable

Before running training:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 9. Close Other Processes

- Close other Python processes using GPU
- Check with: `nvidia-smi`
- Kill processes if needed: `kill <PID>`

## Memory Usage Estimates

**Gemma-7B with 4-bit quantization**:
- Base model: ~4-5 GB
- LoRA adapter: ~100-200 MB
- Training overhead: ~8-12 GB (depends on batch size)
- **Total**: ~12-17 GB

**Mistral-7B with 4-bit quantization**:
- Base model: ~4-5 GB
- LoRA adapter: ~80-200 MB
- Training overhead: ~8-12 GB
- **Total**: ~12-17 GB

## Recommended GPU Memory

- **Minimum**: 16 GB VRAM (with optimizations)
- **Recommended**: 24 GB+ VRAM (for comfortable training)
- **Optimal**: 40 GB+ VRAM (A100, can use larger batches)

## Current Config Settings

### Gemma (Optimized for 16-24GB GPUs)
```yaml
batch_size: 2
gradient_accumulation_steps: 8
max_length: 256
lora_r: 32
no_4bit: false
```

### Mistral (Standard)
```yaml
batch_size: 4
gradient_accumulation_steps: 4
max_length: 512
lora_r: 32
no_4bit: false
```

## If Still Getting OOM

1. **Check actual memory usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Reduce max_length to 192 or 128**

3. **Reduce batch_size to 1**

4. **Increase gradient_accumulation_steps** to maintain effective batch size

5. **Use CPU offloading** (slower but uses less GPU memory):
   - Not currently implemented, but can be added if needed

## Monitoring Memory

During training, monitor with:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

Look for:
- **Memory-Usage**: Should stay below 90% of total
- **GPU-Util**: Should be high (80-100%) during training

