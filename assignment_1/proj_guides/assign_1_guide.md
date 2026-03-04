# Pokémon Image Classification Challenge

This assignment guides you through building models to classify Pokémon images by type (e.g., Water, Fire, Grass) in three progressive stages:

1. **Multilayer Perceptron (MLP) Classification**
2. **Convolutional Neural Network (CNN) Development**
3. **Transfer Learning and Fine-Tuning**

---

## Computational Budget

- **Maximum:** 5 hours (Google Colab/Kaggle free tier)
- **Allocation:**
    - MLP: 1 hour
    - CNN: 1 hour
    - Transfer Learning: 3 hours

**Resource Management Tips:**
- Use early stopping, suitable batch sizes, and model checkpointing.
- Document training time and GPU usage in your presentation.

---

## Dataset

- Pokémon images labeled by primary type.
- **Note:** Class imbalance is present.
- All processing must be feasible on free-tier resources.

Due to the imbalance in the data distribution, submissions will be scored with Macro-averaged F1 Score.

---
## Submission Guidelines

Your submission must be a CSV file with predictions for each Id in the test set. The file should include a header and follow this format:

```
Id,label
2,Normal
5,Fire
6,Grass
...
```

## Kaggle Participation Summary
- **Data Usage:** Use the data only for this competition; do not share, redistribute, or use outside Kaggle.
- **Submission Limits:** Up to 5 submissions per day; select up to 2 final submissions for judging.
- **Data Security:** Keep the dataset secure and do not share with non-participants.
- **License:** Data is subject to competition rules and terms.



## Tasks

### 1. Multilayer Perceptron (MLP) Classification (Weeks 1–2)

**Objective:** Implement an MLP for Pokémon type classification.

**Steps:**
- Data exploration: Analyze label distribution and class imbalance.
- Model development: Design/train an MLP for image data.
- Evaluation: Use metrics that account for imbalance.

**Deliverables:**
- MLP code (Colab/Kaggle notebook link).
- Brief presentation:
    - Data exploration insights
    - Model architecture/rationale
    - Performance metrics & interpretation

---

### 2. Convolutional Neural Network (CNN) Development (Week 3)

**Objective:** Build a CNN to improve classification.

**Steps:**
- Model design: Construct a CNN for the dataset.
- Training: Address class imbalance and overfitting.
- Comparison: Evaluate vs. MLP.

**Deliverables:**
- CNN code (Colab/Kaggle notebook link).
- Comparative analysis presentation:
    - MLP vs. CNN architecture differences
    - Performance improvements/challenges
    - Strategies for imbalance/overfitting

---

### 3. Transfer Learning and Fine-Tuning (Week 4)

**Objective:** Use pre-trained models to boost accuracy.

**Steps:**
- Model selection: Choose a pre-trained model (e.g., ResNet, VGG).
- Fine-tuning: Adapt to Pokémon dataset.
- Enhancements: Apply data augmentation and regularization.

**Deliverables:**
- Fine-tuned model code (Colab/Kaggle notebook link).
- Comprehensive final presentation:
    - Transfer learning process/benefits
    - Impact of augmentation/regularization
    - Final metrics & improvement areas

---

## Peer Review

- Review two groups after each stage.
- Use assignment rubric for evaluation.
- Provide constructive, actionable feedback.
- Peer review counts for **6%** of Assignment 1 grade (2% per stage).

---




## Rubrics

### MLP Classification Report

**Data Exploration and Analysis (6%)**: Provides comprehensive statistical summaries and visualizations; quantifies class imbalances with precise percentages.

**Model Development (10%)**: Designs an MLP with a clear rationale for architecture, activation functions, and regularization techniques; justifies choices based on dataset characteristics.

**Training Efficiency (5%)**: Training is well within time budget; optimizes computational resources effectively.

**Performance Evaluation (10%)**: Conducts comprehensive evaluation using multiple metrics; provides in-depth interpretation of results.

**Presentation Quality (3%)**: Delivers a clear, concise, and well-structured presentation; effectively communicates insights, methodologies, and findings; utilizes visual aids proficiently.

**Peer Review (1%)**: Actively participates in peer review; delivers detailed, constructive feedback with actionable suggestions; reflects on peer feedback to improve own work.

---

### CNN Development Report

**Architecture Design (6%)**: Designs an innovative CNN architecture tailored to the dataset; justifies design choices with relevant research or empirical evidence.

**Training & Optimization (10%)**: Training is well within time budget; employs advanced techniques to effectively address class imbalance and overfitting.

**Performance Evaluation (10%)**: Evaluates CNN comprehensively using multiple metrics; provides insightful analysis of strengths and weaknesses compared to MLP.

**Presentation Quality (3%)**: Delivers a detailed and insightful comparative analysis; effectively highlights differences between MLP and CNN architectures; discusses performance improvements and challenges thoroughly.

**Peer Review (1%)**: Actively participates in peer review; delivers detailed, constructive feedback with actionable suggestions; reflects on peer feedback to improve own work.

---

### Transfer Learning & Fine-Tuning Report

**Model Selection & Justification (6%)**: Selects a highly relevant pre-trained model; provides comprehensive justification based on model architecture, dataset similarity, and expected performance benefits.

**Fine-Tuning & Adaptation (10%)**: Effectively fine-tunes the pre-trained model; applies advanced techniques to adapt it to the dataset, resulting in significant performance improvement.

**Data Augmentation & Regularization (5%)**: Implements a comprehensive range of data augmentation and regularization methods; demonstrates measurable improvements in model performance.

**Performance Evaluation & Interpretation (10%)**: Evaluates the fine-tuned model comprehensively using multiple metrics; provides a thorough interpretation of results, highlighting the impact of transfer learning and fine-tuning on performance.

**Presentation Quality (3%)**: Delivers a clear, concise, and well-structured presentation; effectively communicates methodologies, findings, and insights; utilizes visual aids proficiently to enhance understanding.

**Peer Review (1%)**: Actively participates in peer review; delivers detailed, constructive feedback with actionable suggestions; reflects on peer feedback to improve own work.

## Grading Formula

Let **AG** be the assignment grade:

```math
AG = \text{round}\left(20 \cdot \min\{1,G\},1\text{ digit} \right)
```

```math
G = \sum_{i \in \text{Grading Elements}} \text{weight}_i \cdot \frac{\text{points}_i }{4}
```

- Peer Review grade is **individual** and assigned in the final round.

