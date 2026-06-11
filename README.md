# 🖐️ Two-Stream Sign Language Recognition System

## Research Overview

I developed a PhD-level sign language recognition system as independent research during my bachelor studies. The system uses a novel two-stream architecture combining RGB video and pose keypoints.

## Key Results

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 12.00% |
| Top-5 Accuracy | 50.00% |
| Code Size | 2,000+ lines |
| Inference Speed | <100ms |

## Architecture

- **RGB Stream**: Color histograms, HOG features, motion analysis
- **Pose Stream**: Joint angles, movement velocities, spatial distances
- **Fusion**: Cross-modal bidirectional attention

## How to Run

```bash
python slr_without_torch_fixed.py
