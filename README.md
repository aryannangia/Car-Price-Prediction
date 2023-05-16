# SC1015 A125 Group 5 - Car Prediction Model

<img width="872" alt="Project Logo" src="https://user-images.githubusercontent.com/20625476/233848712-9181c0f3-91f6-456c-a6c7-c179167244ce.png">
 
 ###### (Queiro Design)

**In order to minimize this repository's storage size, we have 2 supporting files upload through OneDrive cloud storage. Please download from: https://tinyurl.com/4yc5aru2 and add into the Output folder. This repository has also been initialised with Git LFS.**

Dataset Folder:
 - cleaned_used_cars_dataset.csv: cleaned dataset with only 40 variables
 - train.csv: cleaned and preprocessed dataset used for training
 - used_cars_data_large.feather: 300,000 samples of the original dataset
 - used_cars_data_medium.feather: 30,000 samples of the original dataset
 - used_cars_data_small.feather: 3,000 samples of the original dataset

## Members
 - Cheam Zhong Sheng Andrew - EDA, Data Cleaning, Bagging Ensemble Models, Slides, Video Editing
 - Aryan Nangia - Data Extraction & Preparation, Technical Documentation, EDA, Slides
 - Gillbert Susilo Wong - EDA, Grid Search, PCA, Recursive training, Slides, Model Training, Results Analysis
 
Contents 
1. Problem Statement 
2. Data Cleaning + Model Training
3. Data Analysis & Conclusion
4. Improvements
5. Models Used 
6. References

Kaggle link: https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset
 
Our group has chosen to predict the price of cars from a used US cars dataset on Kaggle for our mini project. The data was collected in September 2020 by user Ananaymital (AnanayMital, 2020) and consists of 66 unique variables and 3 million rows. Our team’s objective is to create a model that can make accurate and precise predictions for the price of a car so buyers don’t need to worry about overpayment and always guarantee to buy at a reasonable quote. Buying used cars in Singapore has neber been popular due to the staggering COE prices and wish to impact this market. Our motivation stems from the fact that since car prices are in 10’s of 000s, the impact of not getting value for money is quite significant. Eventually our conclusion will help us make informed decisions when we buy our own cars in the future.
 
### 1. Problem Statement 
Which model is most accurate in predicting the price of a used car?<br>
Are there any specific factors buyers should consider to avoid overpayment?<br>
What features justify the high cost of a car?<br>
How should a buyer decide if a used car is worth buying?<br>
How much should a seller price his car to get it sold?

**Problem Definition: Which features of automobiles are necessary to achieve an efficient market and justify its associative prices?**

### 2. Data Cleaning + Model Training
Any columns with majority null values were dropped. Through label and target encoding, we converted all variables to numeric data types. Lastly, we feature engineered new variables by combining or decomposing two variables. We merged two or more variables based on our correlation heatmap where we compared correlations with Car Price. Our baseline model was the decision tree regressor, and the ensemble models used were XGBoost and Random Forest as mentioned in our group video. To prevent overfitting, we cross-validated using the shuffle split method. To improve our ML model, we implemented grid search analysis to tune the hyper parameters of each trained model.

Grid search improved our XGBoost model’s accuracy by 1% but did not improve random forest and decision tree regressor. The models achieved high performance with around 10% mean absolute percentage error.

### 3. Data Analysis & Conclusion
We created a data pipeline to facilitate training and analysing of our various models as outlined in section 5. We then ran recursive feature extraction (rfe) on the random forest model with 30,000 samples to find the best set of features. The highest $R^2$ score are obtained with these 8 features: PowerRPM, TorqueRPM, Savings Amount, Franchise Name, followed by the feature engineered variables through PCA: Engine & Fuel Specifications, Car Usage, Car Space.
<br><br>To facilitate buying and selling of cars, we should keep the information of each car to these 8 variables for achieving an efficient market.

### 4. Improvements
- Survey Singaporean car buyers on which features are important and accordingly adjust our features.
- Implement deep learning models to better understand complex connections & extract more features.
- Further classify cars into, eg: sports cars & daily use cars, and have different respective models. 

### 5. Models Used
 - Decision Tree
 - Random Forest
 - Gradient Boosting (XGBoost)

### 6. References
AnanayMital. (2020, September 21). US used cars dataset. Kaggle. Retrieved March 10, 2023, from https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset
<br><br>Queiro Design. (n.d.). Free templates | CANVA. Canva. Retrieved April 20, 2023, from https://www.canva.com/templates/ 
