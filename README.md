# Slovak Letter Type Prediction (LSTM)

Project for the **UNS 2025** course.

Author: Oleksandr Kletsko

## Task Description

In a Slovak word, one letter is replaced with the symbol `x`. An LSTM-based model predicts what type the original letter was:

- **krátka samohláska** — short vowel
- **dlhá samohláska** — long vowel
- **spoluhláska** — consonant

## Data

- Source: `data.txt` — a text corpus of Slovak sentences (~15,000 lines).
- Words are extracted from the text (`re.findall` with a regex for the Slovak alphabet), shuffled, and limited to `DATA_LEN = 40000` samples.
- Word length is restricted to the `3–20` character range (`MAX_LEN = 20`).
- For each word, one letter of the target type is randomly chosen and replaced with `x`, with class balancing (`counts`/`target`) applied so all three classes are roughly equally represented.
- Characters are encoded via `char2idx` based on a fixed alphabet (`ALFABET`, including long vowels and `x`) and converted to **one-hot** vectors (`to_categorical`).
- Train/test split: 80/20, stratified by class (`train_test_split`, `stratify=y`).

## Model Architecture

```
Input(MAX_LEN, vocab_size)
LSTM(32)
Dense(16, relu)
Dense(3, softmax)
```

- Optimizer: `adam`
- Loss: `sparse_categorical_crossentropy`
- Metric: `accuracy`

## Training and Evaluation

- The model is trained for 10 epochs, `batch_size=64`, with validation on the test set.
- After training, test accuracy is printed and an `accuracy`/`val_accuracy` curve is plotted per epoch.

## Dataset Size Impact Analysis

A separate experiment trains the model on subsets of different sizes (1000, 2000, 5000, 10000, full training set) for 5 epochs each, then plots accuracy as a function of training set size.

## Demo Mode

An interactive cell lets you enter a word containing `x` (e.g. `Kolxče`, `kaxustnica`, `vxľký`), get the model's prediction, and compare it against the true letter entered manually.

## Libraries Used

- `tensorflow.keras` (LSTM, Dense, Sequential)
- `numpy`
- `scikit-learn` (`train_test_split`)
- `matplotlib`
- `re`

## Project Files

- `slovak-letter-lstm.ipynb` — main notebook with the full pipeline (data → preprocessing → model → training → evaluation → demo → analysis)
- `data.txt` — text corpus used to generate the training data
