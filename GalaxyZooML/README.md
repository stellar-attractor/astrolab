# 🌌 Galaxy Morphology Classification  
### Galaxy Zoo 2 · CNN Baseline Model

This project implements a convolutional neural network (CNN) for galaxy morphology classification using the **Galaxy Zoo 2** dataset.

The pipeline demonstrates a full workflow:
data loading → preprocessing → training → evaluation → visualization.

---

## ⚙️ Configuration

- **Input size:** 128 × 128 (RGB)  
- **Batch size:** 64  
- **Epochs:** 10  
- **Pipeline optimization:** `tf.data.AUTOTUNE`  

---

## 📡 Dataset

- **Source:** Galaxy Zoo 2 (via `tensorflow_datasets`)  
- **Split used:** `train`  
- **Content:** Galaxy images + crowd-sourced morphological vote fractions  

---

## 🧠 Label Construction

Labels are derived from Galaxy Zoo vote fractions (`table1`):

- `smooth`
- `features_or_disk`
- `star_or_artifact`

The class is assigned via **argmax** over these probabilities:

```text
Elliptical  → smooth
Spiral      → features_or_disk
Artifact    → star_or_artifact