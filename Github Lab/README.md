# Github Lab - MLOps (IE-7374) 

This lab focuses on 5 modules, which includes creating a virtual environment, creating a GitHub repository, creating Python files, creating test files using pytest and unittest, and implementing GitHub Actions.

This is an enhanced version of the original [Lab1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1). The core calculator (`fun1`–`fun4`) is preserved, with the following **modifications**:
- Additional arithmetic functions: `divide`, `power`, `square_root`
- Statistical functions: `mean`, `median`, `mode`, `variance`, `std_dev`, `correlation`, `percentile`
- A new **DataProcessor** module for CSV data loading, normalization, and filtering
- A new **LinearRegressionModel** built from scratch using OLS (Ordinary Least Squares)
- A custom **housing price dataset** (`data/housing_data.csv`) with 30 records
- Expanded test suites: **43 pytest** + **26 unittest** tests




## Step 1: Creating a Virtual Environment

In software development, it's crucial to manage project dependencies and isolate your project's environment from the global Python environment. This isolation ensures that your project remains consistent, stable, and free from conflicts with other Python packages or projects. To achieve this, we create a virtual environment dedicated to our project. 
 
To create a virtual environment, follow these steps:

1. Open a Command Prompt or Terminal in the directory where you want to create your project.
2. Choose a name for your virtual environment (e.g "github_lab") and run the appropriate command:
 ```
 python -m venv github_lab
 ```
3. Activate the virtual environment
 ```
 source github_lab/bin/activate   # macOS/Linux
 github_lab\Scripts\activate      # Windows
 ```
After activation, you will see the virtual environment's name in your command prompt or terminal, indicating that you are working within the virtual environment.


## Step 2: Creating a GitHub Repository, Cloning and Folder Structure
Now that we have set up our virtual environment, the next step is to create a GitHub repository for our project and establish a structured folder layout. This organization helps maintain your project's code, data, and tests in an organized manner.

### Fork the Repository: 
Click the "Fork" button at the top right of this [repository](https://github.com/RajivPraveen/Mlops) to create your own copy.

### Creating a GitHub Repository
- Open a web browser and go to GitHub.
- In the upper right corner, click the "+" button and select "New repository."
- Choose a name for your repository.
- Choose the visibility of your repository—either public (visible to everyone) or private (accessible only to selected collaborators)
- Check the "Initialize this repository with a README" box. This will create an initial README file that you can edit to provide project documentation.
- Click the "Create repository" button.

### Cloning the Repository
- Open a Command Prompt or Terminal.
- Navigate to the directory where you want to clone your GitHub repository. This should be the same directory where you created your virtual environment.
- Run the following command to clone your GitHub repository into the current directory:
 ```
 git clone https://github.com/RajivPraveen/Mlops.git
 ```
After running the command, the repository will be cloned, and you'll have a local copy of your GitHub project in your chosen directory.

### Establishing Folder Structure
- Once you have cloned your repository, you can establish a structured folder layout within it. This layout helps organize your project into key directories for code, data, and tests. Create the following subfolders within your repository: 
- **data:** This folder is used for storing project data files or datasets.
- **src:** This folder is where you'll store your project's source code files.
- **test:** This folder is dedicated to unit tests and test scripts for your code.
- **workflows:** This folder stores the GitHub Actions workflow YAML files.
- Create a file named `.gitignore`. This is useful to exclude the virtual environment and other unnecessary files from version control.
- Add the virtual environment folder name inside your gitignore file so that it's not tracked by Git.

The final folder structure looks like this:
```
Github Lab/
├── data/
│   └── housing_data.csv
├── src/
│   ├── __init__.py
│   ├── calculator.py
│   ├── data_processor.py
│   └── model.py
├── test/
│   ├── __init__.py
│   ├── test_pytest.py
│   └── test_unittest.py
├── workflows/
│   ├── pytest_action.yml
│   └── unittest_action.yml
├── README.md
└── requirements.txt
```

### Adding and Pushing Your Project Code to GitHub
Now that we have our virtual environment set up, the GitHub repository created, and the folder structure organized, let's add our project's code and push it to GitHub. 

**Adding Your Project Code** 
- Navigate to your project directory using the Command Prompt or Terminal, where you have the virtual environment and folder structure set up.
- Create and write your Python code or other project files within the specified directories (src, data, etc.) according to your project requirements.
- Once your project files are ready, it's time to add them to Git's staging area. In your project directory, run the following command:
 ```
 git add .
 ```
- This command stages all the changes and new files in your project directory for the next commit.

**Committing Your Changes** 
- After staging your changes, commit them with a meaningful commit message that describes the changes you made. Replace with a descriptive message:
 ```
 git commit -m " "
 ```

**Pushing to GitHub** 
- To push your committed changes to your GitHub repository, use the following command:
 ```
 git push origin main
 ```


## Step 3: Creating calculator.py in src Folder
- In this step, we create a Python script named `calculator.py` within the `src` folder of your project. This script contains a set of mathematical functions designed to perform basic arithmetic operations, **plus enhanced statistical functions**.

### Original Functions (from Lab1)
- `fun1(x, y)` adds two input numbers, x and y.
- `fun2(x, y)` subtracts y from x.
- `fun3(x, y)` multiplies x and y.
- `fun4(x, y, z)` adds three numbers together and returns their sum.

### Enhanced Arithmetic Functions (New)
- `divide(x, y)` divides x by y, with zero-division protection.
- `power(base, exponent)` raises base to the power of exponent.
- `square_root(x)` computes the square root of a non-negative number.

### Statistical Functions (New)
- `mean(data)` computes the arithmetic mean of a list.
- `median(data)` computes the median value.
- `mode(data)` finds the most frequently occurring value(s).
- `variance(data, population)` computes population or sample variance.
- `std_dev(data, population)` computes population or sample standard deviation.
- `correlation(x, y)` computes the Pearson correlation coefficient between two lists.
- `percentile(data, p)` computes the p-th percentile.

All functions include input validation and raise appropriate `ValueError` exceptions for invalid inputs (e.g., empty lists, division by zero, negative square roots).

- To view the code and gain a deeper understanding, please refer to the [calculator.py](https://github.com/RajivPraveen/Mlops/blob/main/Github%20Lab/src/calculator.py) file located under the src folder.

### Creating data_processor.py (New Module)
- In addition to the calculator, we created a `DataProcessor` class in `data_processor.py` that handles CSV data operations:
- `load_csv(filepath)` loads and parses a CSV file into structured data.
- `get_column(name)` and `get_numeric_column(name)` extract column values.
- `column_stats(name)` computes summary statistics (count, mean, min, max, std).
- `normalize_column(name)` applies Min-Max normalization to scale values between 0 and 1.
- `filter_rows(column, condition, value)` filters rows using conditions like `>`, `<`, `>=`, `<=`, `==`, `!=`.
- `shape()`, `head()`, and `describe()` provide quick dataset exploration.

- To view the code, refer to the [data_processor.py](https://github.com/RajivPraveen/Mlops/blob/main/Github%20Lab/src/data_processor.py) file.

### Creating model.py (New Module)
- We also implemented a `LinearRegressionModel` class in `model.py` using the Ordinary Least Squares (OLS) method — no external ML libraries required:
- `fit(x, y)` trains the model by computing the optimal slope and intercept.
- `predict(x)` and `predict_single(x)` generate predictions for new data.
- `evaluate(x, y)` computes evaluation metrics: MAE, MSE, RMSE, and R².
- `save_model(filepath)` and `load_model(filepath)` persist model parameters to JSON.

- To view the code, refer to the [model.py](https://github.com/RajivPraveen/Mlops/blob/main/Github%20Lab/src/model.py) file.

### Dataset: housing_data.csv
- A synthetic housing price dataset with 30 records is included in the `data/` folder. It contains 5 features:

| Column | Description |
|---|---|
| `size_sqft` | House size in square feet (750–2300) |
| `bedrooms` | Number of bedrooms (1–5) |
| `age_years` | Age of the house in years (1–35) |
| `distance_downtown_miles` | Distance from downtown (1.0–11.0) |
| `price_thousands` | Sale price in thousands of dollars (155–525) |

> **Note:** 
Whenever you want to push files to your repository follow this step
[Adding and Pushing Your Project Code to GitHub](#adding-and-pushing-your-project-code-to-github)


## Step 4: Creating tests using Pytest and Unittests
- In this step, we set up unit tests for the functions in our `calculator.py`, `data_processor.py`, and `model.py` scripts using two popular testing frameworks: [pytest](https://docs.pytest.org/en/7.4.x/) and [unittest](https://docs.python.org/3/library/unittest.html). Unit testing ensures that individual components of your code work as expected, helping you catch and fix bugs early in the development process.

**Using Pytest** 
- Installation (if not already installed):
- If you haven't already installed pytest, you can do so using pip:
 ```
 pip install pytest
 ```

### Writing Pytest Tests
- Pytest makes it easy to write tests for your Python code. Tests are written as regular Python functions, and test file names typically start with `test_` or end with `_test.py`.
- To run your Pytest tests, you can use the pytest command followed by the name of the test file or directory containing your tests:
 ```
 pytest test/test_pytest.py -v
 ```
- Pytest automatically discovers test functions based on naming conventions. It searches for functions starting with `test_` or ending with `_test`, and it can discover tests in subdirectories as well. This makes it easy to organize your tests.
- Let's create a test file named `test_pytest.py` within the test folder. This file contains a comprehensive series of test functions covering:
  - **Original calculator tests** (`test_fun1`, `test_fun2`, `test_fun3`, `test_fun4`) — verifying the original Lab1 functions.
  - **Enhanced arithmetic tests** (`test_divide`, `test_power`, `test_square_root`) — testing new arithmetic operations including edge cases like division by zero and negative square roots.
  - **Statistical function tests** (`test_mean`, `test_median`, `test_mode`, `test_variance`, `test_std_dev`, `test_correlation`, `test_percentile`) — validating statistical computations with known values and edge cases.
  - **Data processor tests** — loading CSVs, extracting columns, computing stats, normalization, filtering, and testing against the included housing dataset.
  - **Model tests** — training on known linear data (y = 2x + 1), verifying slope/intercept, predictions, evaluation metrics, and save/load functionality.
- Each test function uses the `assert` statement to validate the expected outcomes. Refer to the file under the test folder for your [reference](https://github.com/RajivPraveen/Mlops/blob/main/Github%20Lab/test/test_pytest.py).
- By running these 43 pytest tests, you can verify that all modules are working correctly.

### Writing Tests with UnitTest
- Unittest allows you to write tests as classes that inherit from the `unittest.TestCase` class. Test methods are identified by their names, which must start with `test_` to be recognized as test cases.
- To run Unittest tests, you typically execute your test script, which should include a call to `unittest.main()` at the end. Here's how you can run the tests:
 ```
 cd "Github Lab"
 python -m unittest test.test_unittest -v
 ```
- Unittest relies on test discovery, which means it will find test methods based on naming conventions, similar to Pytest. Test methods must start with `test_` to be recognized as test cases.
- Unittest provides a variety of assertion methods, such as `assertEqual`, `assertTrue`, `assertFalse`, `assertAlmostEqual`, `assertRaises`, and others, to check conditions in your tests. You can choose the assertion method that best suits your testing needs.
- Let's create a test file named `test_unittest.py` within the test folder. This file contains test classes covering:
  - `TestCalculatorBasic` — tests for the original `fun1`–`fun4` functions.
  - `TestCalculatorEnhanced` — tests for `divide`, `power`, `square_root`, `mean`, `median`, `std_dev`, `variance`, `correlation`, and `percentile`.
  - `TestDataProcessor` — tests for loading the housing dataset, extracting numeric columns, computing stats, normalization, and row filtering.
  - `TestLinearRegressionModel` — tests for model training, slope/intercept verification, predictions, R² evaluation, and error handling for untrained models.
- Each test function uses the `self.assertEqual` and `self.assertAlmostEqual` statements to validate expected outcomes. Refer to the file under the test folder for your [reference](https://github.com/RajivPraveen/Mlops/blob/main/Github%20Lab/test/test_unittest.py).
- By running these 26 unittest tests, you can verify that all modules are working correctly.


## Step 5: Implementing GitHub Actions
- GitHub Actions is a powerful automation and CI/CD (Continuous Integration/Continuous Deployment) platform provided by GitHub. It enables you to automate various workflows and tasks directly within your GitHub repository. GitHub Actions can be used for a wide range of purposes, such as running tests, deploying applications, and automating release processes.

**How GitHub Actions Work:** 

- GitHub Actions work based on events, actions, and triggers:
- **Events:** These are specific activities that occur within your GitHub repository, such as code pushes, pull requests, or issue comments. GitHub Actions can respond to these events.
- **Actions:** Actions are individual tasks or steps that you define in a workflow file. These tasks can be anything from building your code to running tests or deploying your application.
- **Triggers:** Triggers are conditions that cause a workflow to run. They can be based on events (e.g., a new pull request) or scheduled to run at specific times.

**The Purpose of GitHub Actions:** 

- GitHub Actions serves several purposes:
- **Automation:** It automates repetitive tasks, reducing manual effort and ensuring consistency in your development process.
- **Continuous Integration (CI):** It allows you to set up CI pipelines to automatically build, test, and validate your code changes whenever new code is pushed to the repository.
- **Continuous Deployment (CD):** It enables automatic deployment of your application when changes are merged into a specific branch, ensuring a smooth and reliable release process.

### Using Pytest and Unittest with GitHub Actions:
- Integrating Pytest and Unittest with GitHub Actions can significantly improve the quality and reliability of your codebase. Here's how:
- **Pytest with GitHub Actions:** You can create a GitHub Actions workflow (e.g., `pytest_action.yml`) that specifies the steps for running your Pytest tests. When events like code pushes or pull requests occur, GitHub Actions will automatically trigger the workflow, running your Pytest tests and reporting the results back to you. This helps you catch bugs and ensure that your code meets quality standards early in the development process.
- **Unittest with GitHub Actions:** Similarly, you can create a separate GitHub Actions workflow (e.g., `unittest_action.yml`) to run your Unittest tests. This ensures that both your Pytest and Unittest suites are executed automatically whenever code changes are made or pull requests are submitted. It provides a robust validation mechanism for your codebase.
- When collaborating in teams, the automated testing process ensures that all test cases pass successfully before allowing the merge of a pull request into the main branch.

### Creating GitHub Actions Workflow Files:
- To implement Pytest and Unittest with GitHub Actions, you'll create two workflow files under the `.github/workflows` directory in your repository: `pytest_action.yml` and `unittest_action.yml`. These workflow files define the specific actions and triggers for running your tests.

**pytest_action.yml** 
Please refer [this](https://github.com/RajivPraveen/Mlops/blob/main/Github%20Lab/workflows/pytest_action.yml) file for your reference
1. Workflow Name: The workflow is named "Testing with Pytest."
2. Event Trigger: It specifies the event that triggers the workflow. In this case, it triggers when code is pushed to the main branch or a pull request is opened.
3. Jobs: The workflow contains a single job named "build," which runs on the ubuntu-latest virtual machine environment across Python versions 3.9, 3.10, 3.11, and 3.12 using a matrix strategy.
4. Steps:
 - Checkout code: This step checks out the code from the repository using actions/checkout@v4.
 - Set up Python: It sets up the Python environment using actions/setup-python@v5 with the matrix Python version.
 - Install Dependencies: This step installs the project dependencies by running `pip install -r "Github Lab/requirements.txt"`.
 - Lint with flake8: Runs flake8 to check for syntax errors and code quality issues.
 - Run Tests and Generate XML Report: The core testing step runs Pytest with the `--junitxml` flag to generate an XML report named `pytest-report.xml`. The `continue-on-error: false` setting ensures that the workflow will be marked as failed if any test fails.
 - Upload Test Results: In this step, the generated XML report is uploaded as an artifact using actions/upload-artifact@v4.
 - Notify on Success and Failure: These two steps use conditional logic to notify based on the outcome of the tests.
   - `if: success()` checks if the tests passed successfully and runs the "Tests passed successfully" message.
   - `if: failure()` checks if the tests failed and runs the "Tests failed" message.

**unittest_action.yml** 
Please refer [this](https://github.com/RajivPraveen/Mlops/blob/main/Github%20Lab/workflows/unittest_action.yml) file for your reference
1. Workflow Name: This GitHub Actions workflow is named "Python Unittests."
2. Event Trigger: The workflow is triggered by the "push" event, specifically when changes are pushed to the main branch, or when a pull request targets main.
3. Jobs: The workflow defines a single job named "build" that runs on the ubuntu-latest virtual machine environment.
4. Steps:
 - Checkout code: This step uses the actions/checkout@v4 action to check out the code from the repository. It ensures that the workflow has access to the latest code.
 - Set up Python: The "Set up Python" step uses the actions/setup-python@v5 action to configure the Python environment. It specifies that Python version 3.11 should be used.
 - Install Dependencies: This step runs the command `pip install -r "Github Lab/requirements.txt"` to install the project's Python dependencies.
 - Run unittests: In this step, the unittest tests are executed using the command `python -m unittest test.test_unittest`. It runs the unittest test suite defined in the `test.test_unittest` module.
 - Notify on success: This step uses conditional logic with `if: success()` to check if all the unittest tests passed successfully. If they did, it runs the message "Unit tests passed successfully."
 - Notify on failure: Similarly, this step uses conditional logic with `if: failure()` to check if any of the unittest tests failed. If any test failed, it runs the message "Unit tests failed."
