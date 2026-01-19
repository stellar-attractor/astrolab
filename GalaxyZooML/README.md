
	1.	Set hyperparams
Uses 128×128 RGB inputs, batch size 64, 10 epochs, tf.data.AUTOTUNE for pipeline performance.

	2.	Preprocessing & labels

	•	Loads Galaxy Zoo 2 from tensorflow_datasets (split="train").
	•	Resizes images to 128×128 and normalizes to [0,1].
	•	Builds a 3‑class label from Galaxy Zoo vote fractions in table1, taking the argmax over:
	•	smooth vs
	•	features_or_disk vs
	•	star_or_artifact.
Thus classes map to Elliptical / Spiral / Artifact.

	3.	Dataset split & input pipeline

	•	Counts total records (with a tqdm loop), then splits 80/20 via take/skip.
	•	Creates efficient pipelines: map(preprocess) → shuffle(1000) → batch(64) → prefetch(AUTOTUNE) for train and test.

	4.	Model
A small CNN:

Conv2D(32,3) → MaxPool → Conv2D(64,3) → MaxPool → Flatten → Dense(64) → Dense(3, softmax)

Compiled with Adam, sparse_categorical_crossentropy, tracking accuracy.
	5.	Training & evaluation

	•	Trains for 10 epochs with validation on the test set.
	•	Prints test accuracy/loss.
	•	Plots loss and accuracy curves for train/val.

	6.	Qualitative results

	•	Runs inference on one test batch.
	•	Shows a 3×3 grid of images with Predicted vs True labels.

⸻

## Notes / caveats (helpful context):
	•	Counting the dataset by iterating over ds_full is slow; TFDS supports splits like "train[:80%]", "train[80%:]" to avoid a full pass.
	•	No handling of class imbalance or data augmentation—this is a baseline.
	•	Validation is run on the same hold‑out as test; for stricter evaluation, keep a separate validation split.
	•	Labels are derived from crowd vote fractions (noisy but useful); you might threshold or weight by vote confidence in future runs.

    Links:
    https://github.com/zachaa/Galaxy_Morphology_Classification