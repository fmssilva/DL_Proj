# F-Score (F-Measure)

## Overview

The **F-score** or **F-measure** is a measure of predictive performance used in statistical analysis of binary classification and information retrieval systems. It is calculated from **precision** and **recall**:

- **Precision** (Positive Predictive Value): `TP / (TP + FP)`
- **Recall** (Sensitivity): `TP / (TP + FN)`

The **F1 score** is the harmonic mean of precision and recall, symmetrically representing both in one metric. The more generic **Fβ score** applies additional weights, valuing one of precision or recall more than the other.

- **Highest possible value:** `1.0` (perfect precision and recall)
- **Lowest possible value:** `0` (if either precision or recall is zero)

---

## Etymology

The name *F-measure* is believed to be named after a different F function in Van Rijsbergen's book, when introduced to the **Fourth Message Understanding Conference (MUC-4, 1992)**.

---

## Definition

### F1 Score

The traditional F-measure or balanced F-score is the **harmonic mean of precision and recall**:

$$F_1 = \frac{2}{\text{recall}^{-1} + \text{precision}^{-1}} = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = \frac{2\,\text{TP}}{2\,\text{TP} + \text{FP} + \text{FN}}$$

**Special cases:**

- If `FP = FN`:
$$F_1 = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{TP}}{\text{TP} + \text{FN}} \implies F_1 = \text{precision} = \text{recall}$$

- If `TP = FP = FN`:
$$F_1 = \frac{2\,\text{TP}}{4\,\text{TP}} = 0.5$$

> **Note:** As a harmonic mean: $F_1^{-1} = \frac{1}{2}(\text{recall}^{-1} + \text{precision}^{-1})$

---

### Fβ Score

A more general score using a positive real factor **β**, where β is chosen such that recall is considered β times as important as precision:

$$F_\beta = \frac{\beta^2 + 1}{(\beta^2 \cdot \text{recall}^{-1}) + \text{precision}^{-1}} = \frac{(1 + \beta^2) \cdot \text{precision} \cdot \text{recall}}{(\beta^2 \cdot \text{precision}) + \text{recall}}$$

In terms of Type I and Type II errors:

$$F_\beta = \frac{(1 + \beta^2) \cdot \text{TP}}{(1 + \beta^2) \cdot \text{TP} + \beta^2 \cdot \text{FN} + \text{FP}}$$

**Common β values:**
| β     | Effect                              |
| ----- | ----------------------------------- |
| `2`   | Weighs recall higher than precision |
| `0.5` | Weighs precision higher than recall |

The Fβ score is related to Van Rijsbergen's effectiveness measure:

$$E = 1 - \left(\frac{\alpha}{p} + \frac{1-\alpha}{r}\right)^{-1}$$

where $F_\beta = 1 - E$ and $\alpha = \frac{1}{1 + \beta^2}$.

---

## Diagnostic Testing Reference Table

|                       | **Predicted Positive**             | **Predicted Negative**              |
| --------------------- | ---------------------------------- | ----------------------------------- |
| **Real Positive (P)** | True Positive (TP)                 | False Negative (FN) — Type II error |
| **Real Negative (N)** | False Positive (FP) — Type I error | True Negative (TN)                  |

**Derived metrics:**

| Metric                                          | Formula                                               |
| ----------------------------------------------- | ----------------------------------------------------- |
| True Positive Rate (TPR) / Recall / Sensitivity | `TP / P`                                              |
| False Negative Rate (FNR)                       | `FN / P = 1 − TPR`                                    |
| False Positive Rate (FPR)                       | `FP / N = 1 − TNR`                                    |
| True Negative Rate (TNR) / Specificity          | `TN / N`                                              |
| Precision / PPV                                 | `TP / (TP + FP)`                                      |
| False Discovery Rate (FDR)                      | `FP / (TP + FP) = 1 − PPV`                            |
| Negative Predictive Value (NPV)                 | `TN / (TN + FN)`                                      |
| False Omission Rate (FOR)                       | `FN / (TN + FN) = 1 − NPV`                            |
| Accuracy (ACC)                                  | `(TP + TN) / (P + N)`                                 |
| Balanced Accuracy (BA)                          | `(TPR + TNR) / 2`                                     |
| F1 Score                                        | `2TP / (2TP + FP + FN)`                               |
| Fowlkes–Mallows Index (FM)                      | `√(PPV × TPR)`                                        |
| Matthews Correlation Coefficient (MCC)          | `√(TPR × TNR × PPV × NPV) − √(FNR × FPR × FOR × FDR)` |
| Threat Score (TS) / Jaccard Index               | `TP / (TP + FN + FP)`                                 |

---

## Dependence on Class Imbalance

The precision-recall curve, and thus Fβ, **explicitly depends on the ratio of positive to negative cases**. This makes cross-problem comparison problematic when class ratios differ. One approach is to use a standard class ratio `r₀` for such comparisons.

---

## Applications

The F-score is widely used in:

- **Information retrieval:** search, document classification, query classification
- **Machine learning:** classification performance evaluation
- **Natural language processing:** named entity recognition, word segmentation

> **Note:** F-measures do not account for true negatives. Metrics such as **MCC**, **Informedness**, or **Cohen's kappa** may be preferred for binary classifier assessment.

---

## Extension to Multi-Class Classification

### Macro F1
Macro F1 is a **macro-averaged F1 score** for balanced performance measurement. Two formulas are used:
1. F1 of class-wise arithmetic mean of precision and recall
2. **Arithmetic mean of class-wise F1 scores** *(preferred — more desirable properties)*

### Micro F1
Micro F1 is the **harmonic mean of micro precision and micro recall**.
- In single-label multi-class classification: `micro precision = micro recall = micro F1`
- Micro F1 ≠ accuracy in general (accuracy accounts for true negatives; micro F1 does not)

---

## Properties

- F1 score is the **Dice coefficient** of retrieved and relevant item sets.
- A classifier always predicting the positive class has F1 = `2p / (1 + p)`, where `p` is the proportion of the positive class.
- If the model is uninformative, the optimal threshold is `0` (always predict positive).
- F1 score is **concave** in the true positive rate.

---

## Criticism

- **David Hand:** F1 gives equal importance to precision and recall, ignoring that different misclassification costs may apply in practice.
- **Chicco & Jurman:** MCC is more truthful and informative than F1 in binary classification.
- **David M W Powers:** F1 ignores true negatives, making it misleading for unbalanced classes. Proposes **Informedness** and **Markedness** as alternatives.
- **Lack of symmetry:** F1 may change value when dataset labels are swapped. The **P4 metric** addresses this as a symmetric extension.
- **Ferrer; Dyrland et al.:** Expected cost/utility is a more principled metric, showing F1 can lead to wrong conclusions.

---

## Difference from Fowlkes–Mallows Index

| Metric                | Aggregation Method                     |
| --------------------- | -------------------------------------- |
| F-measure             | Harmonic mean of recall and precision  |
| Fowlkes–Mallows Index | Geometric mean of recall and precision |

---

## See Also

- BLEU, METEOR, ROUGE, NIST, LEPOR, Word Error Rate
- Confusion Matrix
- Receiver Operating Characteristic (ROC)
- Uncertainty Coefficient

---

## References

1. Sasaki, Y. (2007). *The truth of the F-measure*. Teach Tutor Mater, 1(5), 1–5.
2. Aziz Taha, A. (2015). Metrics for evaluating 3D medical image segmentation. *BMC Medical Imaging*, 15(29). doi:10.1186/s12880-015-0068-x
3. Van Rijsbergen, C. J. (1979). *Information Retrieval* (2nd ed.). Butterworth-Heinemann.
4. Fawcett, T. (2006). An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861–874.
5. Powers, D. M. W. (2011). Evaluation: From Precision, Recall and F-Measure to ROC. *Journal of Machine Learning Technologies*, 2(1), 37–63.
6. Chicco D, Jurman G (2020). The advantages of the MCC over F1 score. *BMC Genomics*, 21(1). doi:10.1186/s12864-019-6413-7
7. Opitz, J. (2024). A Closer Look at Classification Evaluation Metrics. *Transactions of the Association for Computational Linguistics*, 12, 820–836.
8. Opitz, J.; Burst, S. (2019). Macro F1 and Macro F1. arXiv:1911.03347
9. Brownlee, J. (2021). *Imbalanced Classification with Python*. Machine Learning Mastery. ISBN 979-8468452240.

---

*Content adapted from Wikipedia under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Last edited 21 January 2026.*
