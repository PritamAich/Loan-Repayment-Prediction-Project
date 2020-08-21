# Loan-Repayment-Prediction-Project

## Project Overview:

#### Intro

LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.

#### Our Goal

Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), we want to build a model that can predict wether or not a borrower will pay back their loan.

The "loan_status" column contains our label or it is the dependent variable(predicting class).

## Resources:

**Python version :** 3.7
**Packages Used:** pandas, numpy, matplotlib, seaborn, sklearn, tensorflow.

### Exploratory Data Analysis:

1. Used count plot to measure if the data is balanced or not.
2. Plotted distribution plots, scatter plots, count plots , box plots, boxen plots to access the behaviour of different attributes.
3. These plots helped in gaining some meaningful insights regarding customer behaviours towards paying back the loan.

### Data Preprocessing and Feature Engineering:

1. Removed some of the features that have practically no impact of predicting if a burrower will pay back a loan.
2. Removed some null values that are very few in numbers(less than 0.5% of our data)
3. Created dummy variables for some data categories.
3. Did some Feature engineering on categorical data(like extracting the year from a date, zip code from address column).

### Model Building:

1. First splitted the data that was retrieved after preprocessing. The data was split into traing , validation and test sets.
2. Then inputs from all three data sets were sclaled using Standard Scaler.
3. The scaled data is saved in .npz format for the final model.
4. Built a basic Artificial Neural Network(ANN) with % hidden layers. Used Softmax activation function for the output layer.
5. Achieved an overall accuracy of 84%.
6. Since this is an imbalanced dataset, the minority class had an f1-score of around 64%, precision around 58% and Recall of around 70%.
                
