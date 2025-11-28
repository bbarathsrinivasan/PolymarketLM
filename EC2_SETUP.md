# EC2 Setup Guide for Polymarket LLM Fine-Tuning

This guide walks you through setting up an EC2 instance for fine-tuning the Polymarket LLM.

## Instance Selection

### Recommended Instance Types

| Instance Type | GPU | VRAM | vCPUs | RAM | Cost/Hour* | Best For |
|--------------|-----|------|-------|-----|------------|----------|
| g5.2xlarge | 1x A10G | 24GB | 8 | 32GB | ~$1.00 | Full training |
| g5.4xlarge | 1x A10G | 24GB | 16 | 64GB | ~$2.00 | Faster training |
| p3.2xlarge | 1x V100 | 16GB | 8 | 61GB | ~$3.00 | Alternative option |
| g4dn.xlarge | 1x T4 | 16GB | 4 | 16GB | ~$0.50 | Budget option |

*Approximate on-demand pricing. Spot instances are 60-70% cheaper.

**Recommendation**: Start with `g5.2xlarge` for a good balance of performance and cost.

## Step 1: Launch EC2 Instance

1. **Go to EC2 Console** → Launch Instance

2. **Configure Instance**:
   - **Name**: `polymarket-llm-training`
   - **AMI**: Choose **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4.1 (Ubuntu 22.04) 20250302**
     - These come with CUDA, cuDNN, and Python pre-installed
     - Search for "Deep Learning" in AMI marketplace
   - **Instance Type**: `g5.2xlarge` (or your preferred type)
   - **Key Pair**: Create or select an existing key pair for SSH access
   - **Storage**: 100GB+ (model + dataset + checkpoints)

3. **Configure Security Group**:
   - Allow SSH (port 22) from your IP
   - Optionally allow HTTP (port 80) for Jupyter/tensorboard

4. **Launch Instance**

## Step 2: Connect to Instance

```bash
# Replace with your key file and instance IP
ssh -i your-key.pem ubuntu@<instance-public-ip>
```

## Step 3: Environment Setup

### 3.1 Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 3.2 Verify CUDA

```bash
nvidia-smi
```

You should see GPU information. If not, the AMI may need CUDA drivers installed.

### 3.3 Setup Python Environment

```bash
# Navigate to project directory (or clone your repo)
cd ~
git clone <your-repo-url> polymarket-llm
cd polymarket-llm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3.4 Setup Data

**Option A: Upload from local machine**:
```bash
# On your local machine
scp -i your-key.pem data/fine_tune.jsonl ubuntu@<instance-ip>:~/polymarket-llm/data/
```

**Option B: Download from S3**:
```bash
# On EC2 instance
aws s3 cp s3://your-bucket/fine_tune.jsonl data/
```

**Option C: Generate on EC2** (if you have raw data):
```bash
python scripts/preprocess_data.py
```

## Step 4: Training Configuration

### 4.1 Quick Test (Recommended First)

Test the setup with a small sample:

```bash
# Create sample dataset
python scripts/create_sample_dataset.py --num_samples 50

# Quick training test
python scripts/finetune_qlora.py \
    --dataset_path data/fine_tune_sample.jsonl \
    --max_samples 50 \
    --num_epochs 1 \
    --batch_size 2 \
    --save_steps 10 \
    --logging_steps 5
```

This verifies everything works before committing to full training.

### 4.2 Full Training

Once the test passes, run full training:

```bash
# Run training in background with nohup
nohup python scripts/finetune_qlora.py \
    --dataset_path data/fine_tune.jsonl \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --output_dir models/checkpoints \
    --save_steps 200 \
    --logging_steps 50 \
    > training.log 2>&1 &

# Monitor training
tail -f training.log
```

### 4.3 Using Screen/Tmux (Recommended)

For better session management:

```bash
# Install screen
sudo apt-get install screen -y

# Start screen session
screen -S training

# Run training
python scripts/finetune_qlora.py \
    --dataset_path data/fine_tune.jsonl \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --output_dir models/checkpoints

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

## Step 5: Monitoring Training

### 5.1 Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi
```

### 5.2 Monitor Training Logs

```bash
# If using nohup
tail -f training.log

# If using screen
screen -r training
```

### 5.3 Check Disk Space

```bash
df -h
```

Ensure you have enough space for checkpoints.

## Step 6: Save Results

### 6.1 Download Checkpoints

```bash
# On your local machine
scp -i your-key.pem -r ubuntu@<instance-ip>:~/polymarket-llm/models/checkpoints/ ./
```

### 6.2 Upload to S3 (Recommended)

```bash
# On EC2 instance
aws s3 sync models/checkpoints/ s3://your-bucket/polymarket-llm/checkpoints/
```

## Step 7: Cost Optimization

### 7.1 Use Spot Instances

Spot instances are 60-70% cheaper:
- Launch a spot instance with the same configuration
- Set max price (e.g., $0.50/hour for g5.2xlarge)
- Use checkpointing to resume if interrupted

### 7.2 Stop Instance When Not Training

```bash
# Stop instance (saves money, keeps data)
# Via AWS Console or CLI
aws ec2 stop-instances --instance-ids <instance-id>
```

### 7.3 Estimate Costs

- **g5.2xlarge**: ~$1.00/hour × 4 hours = $4.00 for full training
- **Spot instance**: ~$0.30/hour × 4 hours = $1.20

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# If not working, install drivers
sudo apt-get install nvidia-driver-470 -y
sudo reboot
```

### Out of Memory

Reduce batch size or sequence length:
```bash
python scripts/finetune_qlora.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 256
```

### Training Interrupted

Resume from checkpoint:
```bash
# TrainingArguments automatically resumes from latest checkpoint
# Just run the same command again
```

### Connection Lost

Use screen/tmux to keep session alive:
```bash
screen -r training  # Reattach to session
```

## Best Practices

1. **Always test locally first** with a small sample
2. **Use screen/tmux** to keep training running if connection drops
3. **Monitor GPU usage** to ensure efficient utilization
4. **Save checkpoints frequently** (every 200 steps)
5. **Upload results to S3** before terminating instance
6. **Use spot instances** for cost savings (with checkpointing)
7. **Stop instance** when not in use to save costs

## Cleanup

After training is complete:

```bash
# Download results
scp -i your-key.pem -r ubuntu@<instance-ip>:~/polymarket-llm/models/ ./

# Terminate instance (via AWS Console or CLI)
aws ec2 terminate-instances --instance-ids <instance-id>
```

**Note**: Terminating deletes all data. Make sure to download/upload results first!

## Additional Resources

- [AWS EC2 GPU Instances](https://aws.amazon.com/ec2/instance-types/)
- [Deep Learning AMI Documentation](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
- [Spot Instance Best Practices](https://aws.amazon.com/ec2/spot/getting-started/)

