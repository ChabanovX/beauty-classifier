# SCUT-FBP5500 dataset

Dataset is in `scut/` directory.

Cleaned up dataset structure, renamed, and/or removed unnecessary files by [me](https://github.com/sasaSilver).

- `images` directory contains 5500 frontal, unoccluded faces (350 x 350px) aged from 15 to 60 with neutral expression. It can be divided into four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males.

- `.txt` files in `labels/raw` directory contain labeled attractiveness (1.0 - 5.0) for each face file in the dataset.

- `.txt` files in `labels/processed` directory contain labeled attractiveness (0.0 - 1.0) for each face file in the dataset with columns `name` and `score`.

- `test.txt` -- the test set (60% of the dataset).

- `train.txt` -- the training set (40% of the dataset).

- `all.txt` -- the full dataset (100% of the dataset).