# Iris Flower Classification

A Python script that classifies iris flowers into one of three species (*Iris setosa*, *Iris versicolor*, *Iris virginica*). It bases the prediction on four flower measurements and uses a k-nearest neighbors (KNN) model from scikit-learn.

## Dataset

The script downloads the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) directly from the UCI Machine Learning Repository, so there are no data files to set up. The dataset has 150 samples, 50 per species, with these columns:

| Column         | Description                  |
|----------------|------------------------------|
| `sepal_length` | Sepal length (cm)            |
| `sepal_width`  | Sepal width (cm)             |
| `petal_length` | Petal length (cm)            |
| `petal_width`  | Petal width (cm)             |
| `class`        | Species (the target to predict) |

## What the script does

1. **Loads the data** from the UCI URL into a pandas DataFrame.
2. **Visualizes it** with a seaborn pair plot, which shows each pair of features and colors the points by species.
3. **Splits the data** into 70% training and 30% testing (`random_state=42`, so the split is the same on every run).
4. **Trains a KNN classifier** with `k = 3` neighbors.
5. **Evaluates the model** on the test set and prints the accuracy and a full classification report (precision, recall, F1-score).
6. **Predicts a new flower** from measurements you supply (sepal 5.7 × 4.5 cm, petal 3.4 × 0.9 cm in the example).

## Requirements

- Python 3.x (built and tested with Python 3.14)
- pandas
- matplotlib
- seaborn
- scikit-learn

Install the libraries with:

```
python -m pip install pandas matplotlib seaborn scikit-learn
```

The first run needs an internet connection to download the dataset.

## How to run

From the `IrisFlowerRecognition` folder:

```
python IrisFlowerClassification.py
```

A pair plot window opens first. **Close it to let the script continue**, after which the results print in the terminal.

## Example output

```
Accuracy: 1.0
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        19
Iris-versicolor       1.00      1.00      1.00        13
 Iris-virginica       1.00      1.00      1.00        13

       accuracy                           1.00        45
      macro avg       1.00      1.00      1.00        45
   weighted avg       1.00      1.00      1.00        45

['Iris-versicolor']
```

The model classifies all 45 test flowers correctly and predicts *Iris versicolor* for the example flower. The script also prints the full feature table and the species labels before these results.

## Predicting your own flower

Edit the values in `new_data` near the bottom of the script:

```python
new_data = pd.DataFrame({"sepal_length": [5.7], "sepal_width": [4.5],
                         "petal_length": [3.4], "petal_width": [0.9]})
```

To predict several flowers at once, put more values in each list.

## Troubleshooting

**`ModuleNotFoundError: No module named 'pandas'`**: the Python running the script doesn't have the libraries installed. In VS Code, press `Ctrl+Shift+P` → **Python: Select Interpreter** and choose the Python you installed the libraries into. Then run the install command above using that same Python.
"# IrisFlowerClassification" 
