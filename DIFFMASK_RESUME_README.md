# DiffMask Training Resume Guide

This guide explains how to resume DiffMask training from checkpoints to continue training from where you left off.

## Available Scripts

### 1. `check_checkpoints.sh`
Lists all available checkpoints in the DiffMask model folder.
```bash
./check_checkpoints.sh
```

### 2. `diffmask_resume_latest.sh`
Automatically finds the latest checkpoint and resumes training to reach the target step count.
```bash
./diffmask_resume_latest.sh
```

### 3. `diffmask_resume_train.sh`
Resumes training from a specific checkpoint file. You need to modify the `checkpoint_path` variable.
```bash
./diffmask_resume_train.sh
```

## How to Resume Training

### Option 1: Auto-resume from Latest Checkpoint (Recommended)

1. **Check available checkpoints:**
   ```bash
   ./check_checkpoints.sh
   ```

2. **Modify the target steps in `diffmask_resume_latest.sh`:**
   ```bash
   # Edit this line in diffmask_resume_latest.sh
   target_steps=80001  # Change to your desired total steps
   ```

3. **Run the resume script:**
   ```bash
   ./diffmask_resume_latest.sh
   ```

### Option 2: Resume from Specific Checkpoint

1. **Check available checkpoints:**
   ```bash
   ./check_checkpoints.sh
   ```

2. **Modify `diffmask_resume_train.sh`:**
   ```bash
   # Edit these lines in diffmask_resume_train.sh
   train_num_steps=80001  # Your target total steps
   checkpoint_path="DiffMask/DiffMask_Model/diffmask.pt"  # Path to your checkpoint
   ```

3. **Run the resume script:**
   ```bash
   ./diffmask_resume_train.sh
   ```

## Example: Resume from Step 15,000 to Step 80,000

If you stopped training at step 15,000 and want to continue to step 80,000:

1. **Use the auto-resume script:**
   ```bash
   # Edit diffmask_resume_latest.sh
   target_steps=80001  # Set to 80,001 for 80,000 steps
   ```

2. **Run the script:**
   ```bash
   ./diffmask_resume_latest.sh
   ```

The training will automatically:
- Load the latest checkpoint (which contains step 15,000)
- Continue training from step 15,001
- Stop when reaching step 80,000

## Checkpoint File Naming Convention

- `model-{milestone}.pt`: Checkpoint saved every 1000 steps
  - `model-15.pt` = Step 15,000
  - `model-20.pt` = Step 20,000
  - etc.
- `diffmask.pt`: Latest checkpoint (may not follow milestone naming)

## Important Notes

1. **Step Calculation:**
   - Milestone number × 1000 = Actual step number
   - Example: `model-15.pt` = Step 15,000

2. **Training Parameters:**
   - All training parameters (batch size, learning rate, etc.) are preserved from the original training
   - Only the `train_num_steps` parameter determines when to stop

3. **Checkpoint Loading:**
   - The system automatically loads the model state, optimizer state, and EMA model
   - Training continues exactly from where it left off

4. **Monitoring:**
   - Check the console output to see the current step number
   - The training will show: `{step}: {loss}`

## Troubleshooting

### No checkpoints found:
```bash
./check_checkpoints.sh
```
If no checkpoints are found, you need to run the initial training first:
```bash
./diffmask_train.sh
```

### Checkpoint loading error:
- Ensure the checkpoint file path is correct
- Check that the checkpoint file exists and is not corrupted
- Verify that the model architecture matches the checkpoint

### Training stops early:
- Check that `train_num_steps` is set correctly
- Ensure the checkpoint was loaded properly (check console output)

## Example Workflow

```bash
# 1. Check what checkpoints you have
./check_checkpoints.sh

# 2. Resume training to 80,000 steps
./diffmask_resume_latest.sh

# 3. Monitor training progress
# The console will show: 15001: 0.123, 15002: 0.119, etc.

# 4. Training will automatically stop at step 80,000
```

This setup allows you to easily resume training from any checkpoint and continue to your desired step count. 