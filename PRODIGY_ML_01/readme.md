
# Machine Learning Task 01

This project is a machine learning task focused on data analysis and building a regression model using Python. The task involves reading data, performing exploratory data analysis, and applying a linear regression model.

## Prerequisites

Ensure you have the following libraries installed before running the notebook:

- numpy
- pandas
- scikit-learn
- matplotlib

You can install these libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Open the Jupyter Notebook:**

    Launch Jupyter Notebook in your terminal:

    ```bash
    jupyter notebook
    ```

    Then open `ML_TASK_01.ipynb`.

3. **Load the data:**

    Ensure you have the dataset `train.csv` in the same directory as the notebook.

4. **Run the cells:**

    Execute the cells in the notebook sequentially. The first few cells will import necessary libraries and load the dataset.

## Notebook Content

1. **Import Libraries:**

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    ```

2. **Load the Dataset:**

    ```python
    df = pd.read_csv("train.csv")
    df.head()
    df.columns
    ```

3. **Prepare the Data:**

    ```python
    y = df['SalePrice']
    ```

    Further data preparation steps will follow, such as feature selection, handling missing values, etc.

4. **Model Training:**

    Split the data into training and testing sets, train the linear regression model, and evaluate its performance.

5. **Visualization:**

    Plot the results and visualize the performance of the model.

## Contributing

If you wish to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.
