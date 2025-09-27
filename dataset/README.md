# SCUT-FBP5500 dataset

Dataset is in `data/` directory.

Cleaned up dataset structure, renamed, and/or removed unnecessary files by [me](https://github.com/sasaSilver).

1. `images` directory contains 5500 frontal, unoccluded faces (350 x 350px) aged from 15 to 60 with neutral expression. It can be divided into four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males.

2. `.txt` files in `labels` directory contain labeled attractiveness (1.0 - 5.0) for each face in the dataset.

3. `test.txt` -- the test set (60% of the dataset).

4. `train.txt` -- the training set (40% of the dataset).

5. `all.txt` -- the full dataset (100% of the dataset).