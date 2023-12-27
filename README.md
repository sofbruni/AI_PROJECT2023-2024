# Artificial Intelligence & Machine Learning - 2023/2024 : Trains Dataset Project (title to be changed)
-Bruni Sofia        (285231)
-Sebastiani Mattia  (......)
-Torella Marta      (284091)

## A Brief Introduction to our Project (Section 1)
Welcome to our project and get ready to follow us in this journey as we explore the duties of a senior data scientist for the "famous" **ThomasTrain** company! As a matter of fact, our main goal is to help the company to improve its marketing campaign, and consequently to make the customer retention higher, by providing a model able to predict the satisfaction of a customer. To accomplish this task, we were provided with the the trains.cv, a dataset containing both categorical and numerical features about the customers and their satisfaction regarding different aspects of their experience; once we identified the binary categorical variable "Satisfied" (Y/N) as our target, we decided to approach the problem as a classification task. Here are the main steps we followed: 
- **Checking Data Integrity**: we started by checking the dataset for missing values, duplicates and outliers, and then we proceeded to deal with them.
- **Exploratory Data Analysis**: we performed a thorough analysis of the dataset, in order to understand the relationships between the features and the target, and to identify the most important features.
- **Preprocessing**: we encoded the categorical features, and then we standardized the numerical ones.
- **Splitting**: we split the dataset into training and test set.
- **Models Selection and Training**: we trained different models, and we evaluated them based on different metrics.
- **Hyperparameters Tuning**: we tuned the hyperparameters of our models, in order to improve their performance.
- **Final Evaluation**: we evaluated our models on the test set, and we drew our final conclusions. 

## Methods (Section 2)
### Environment
We used Python version 3.11.1, and the following libraries:
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
### Data Overview: trains.csv
In order to train and test our models we were provided with the trains.csv dataset, in which each row represents a different Ticket ID while the columns corresponds to different features according to clients' evaluation of their experience with the compay and their personal information. Among the 25 features composing our dataset, 19 ('Legroom Service Rating', 'Onboard General Rating', "Food'n'Drink Rating", 'Age', 'Departure Arrival Time Rating', 'Onboard WiFi Rating', 'Ticket ID', 'Checkin Rating', 'Track Location Rating', 'Baggage Handling Rating', 'Arrival Delay in Minutes', 'Boarding Rating', 'Online Booking Rating', 'Onboard Entertainment Rating', 'Distance', 'Seat Comfort Rating', 'Departure Delay in Minutes', 'Onboard Service Rating', 'Cleanliness Rating') are numerical, 14 of which are included in the range [0,5] since they represent ratings assigned to different services by costumers, while the remaining 6 ('Ticket Class', 'Loyalty', 'Gender', 'Satisfied', 'Date and Time', 'Work or Leisure') are categorical. The target variable is the binary categorical variable "Satisfied" (Y/N), which we are going to predict. 
### Preparation of Data
We loaded the dataset into a pandas dataframe and proceeded to check for missing values; after we found out that the only feature containing missing values was "Arrival Delay in Minutes", we decided to drop the rows corresponding to those missing values, since they were only 393 out of 129880, and we considered them negligible. We also decided to drop two variables: "Ticket ID", since it is a unique identifier for each row, and "Date and Time", since it is not a feature that could be used to predict the target variable. Then, we encoded the categorical features using the OneHotEncoder only for the 'Ticket Class'variable in order to impose a 'hierarchy' among the different classes, while we used the LabelEncoder for the remaining ones, included the target variable. Finally, we standardized the numerical features using the StandardScaler after splitting the dataset into Training and Test set: we fit the scaler on training data only to avoid data leakage, and then we transformed both the training and the test set.
### Models
