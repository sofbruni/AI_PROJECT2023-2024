# Artificial Intelligence & Machine Learning - 2023/2024 : Trains Dataset Project (title to be changed)
- **Bruni Sofia        (285231)**
- **Sebastiani Mattia  (......)**
- **Torella Marta      (284091)**

## A Brief Introduction to our Project ⚙️ (section 1)
Welcome to our project and get ready to follow us in this journey as we explore the duties of a senior data scientist for the "famous" **ThomasTrain** company! As a matter of fact, our main goal is to help the company to improve its marketing campaign, and consequently to make the customer retention higher, by providing a model able to predict the satisfaction of a customer. To accomplish this task, we were provided with the the trains.cv, a dataset containing both categorical and numerical features about the customers and their satisfaction regarding different aspects of their experience; once we identified the binary categorical variable "Satisfied" (Y/N) as our target, we decided to approach the problem as a classification task. Here are the main steps we followed: 
- **Checking Data Integrity**: we started by checking the dataset for missing values, duplicates and outliers, and then we proceeded to deal with them.
- **Exploratory Data Analysis**: we performed a thorough analysis of the dataset, in order to understand the relationships between the features and the target, and to identify the most important features.
- **Preprocessing**: we encoded the categorical features, and then we standardized the numerical ones.
- **Splitting**: we split the dataset into training and test set.
- **Models Selection and Training**: we trained different models, and we evaluated them based on different metrics.
- **Hyperparameters Tuning**: we tuned the hyperparameters of our models, in order to improve their performance.
- **Final Evaluation**: we evaluated our models on the test set, and we drew our final conclusions. 

## Methods 🕵🏼‍♂️ (section 2)
### Environment
We used Python version 3.11.1, and the following libraries:
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
### Dataset Overview: trains.csv
In order to train and test our models we were provided with the trains.csv dataset, in which each row represents a different Ticket ID while the columns corresponds to different features according to clients' evaluation of their experience with the compay and their personal information. Among the 25 features composing our dataset, 19 ('Legroom Service Rating', 'Onboard General Rating', "Food'n'Drink Rating", 'Age', 'Departure Arrival Time Rating', 'Onboard WiFi Rating', 'Ticket ID', 'Checkin Rating', 'Track Location Rating', 'Baggage Handling Rating', 'Arrival Delay in Minutes', 'Boarding Rating', 'Online Booking Rating', 'Onboard Entertainment Rating', 'Distance', 'Seat Comfort Rating', 'Departure Delay in Minutes', 'Onboard Service Rating', 'Cleanliness Rating') are numerical, 14 of which are included in the range [0,5] since they represent ratings assigned to different services by costumers, while the remaining 6 ('Ticket Class', 'Loyalty', 'Gender', 'Satisfied', 'Date and Time', 'Work or Leisure') are categorical. The target variable is the binary categorical variable "Satisfied" (Y/N), which we are going to predict. 
### Preparation of Data and Dataset Exploration
We loaded the dataset into a pandas dataframe and proceeded to check for missing values; after we found out that the only feature containing missing values was "Arrival Delay in Minutes", we decided to drop the rows corresponding to those missing values, since there were only 393 missing values out of 129880 total values, and we considered them negligible. We also decided to drop two variables: "Ticket ID" and "Date and Time": as a matter of fact the 'Ticket ID' is a unique ID assigned to the travel ticket not providing any additional information about our customers, while the 'Date and Time' is a variable containing the date and time of the travel that again doesn't provide any information about the satisfaction rate (different from the Departure Arrival Time Rating which instead rates the punctuality of the services provided by the company).

After cleaning the data, we started conducting our EDA (Exploratory Data Analysis) by getting a first visive evaluation of the variables interactions by plotting the correlation matrix for the numerical features, in which we observed that the highest correlation (excluding the values along the diagonal and the correlation between 'Departure Delay in Minutes' and 'Arrival Delay in Minutes' which is quite straightforward) was achieved by 'Online Booking Rating' and 'Onboard Wifi Rating' which can be explained by the fact that both features are related to the online experience of the customer. Other relevant high correlation scores were achieved between 'Cleanliness Rating' and 'Onboard Entertainment Rating', between 'Seat Rating' and 'Cleanliness Rating', and lastly between 'Cleanliness Rating' and 'Onboard Entertainment Rating', which show how customers positively valued a clean environment and its comforts.
![Alt text](images/correlation_matrix_heatmap1.png)

Then, we used some **'countplots'** to visualize the distribution of our target variable 'Satisfied' and of the remaining categorical features; the we formulated the following initial assumptions:

• **Satisfied**: for our target variable we can immediately see that the number of satisfied customers is lower than the number of unsatisfied customers, which could already be an indicator of a general trend in the data that might lead to a slightly unbalanced dataset;

• **Gender**: the presence of male and female customers was almost perfectly balanced in our dataset; 

• **Distribution of Loyalty**: customers that had already joined the company's loyalty program were more likely to use again the services provided by the company rather than customers that were not part of the loyalty program; this shows how the data collected for this analysis mainly came from regular customers who might have experienced the company' services more than once, making their ratings more reliable; 

• **Distribution of Ticket Class**: the purchase levels of Premium and Economy tickets registered almost the same values for both, while Smart Class tickets were only purchased by a small fraction of the total customers; therefore, the company might be facing two different types of customers: the ones that are willing to spend more for a better service (Premium Class) and the ones that are more price-sensitive and are willing to sacrifice some comfort for a cheaper ticket(Economy Class), without having an 'in-between' category of customers that are willing to spend a little more for a better service than the Economy Class but not as much as the Premium Class (Smart Class);

• **Distribution of Work or Leisure**: the majority of the customers were travelling for work rather than for leisure; this could be explained by the fact that the company mainly operates in the business sector, providing services for business travellers rather than for leisure travellers;
![Alt text](images/satisfied_distriburion.png)  
![Alt text](images/cat_distribution.png)

We finally explored the behaviour of our numerical features by plotting their histograms in order to get a better understanding of their distribution and to identify possible outliers; we observed that the majority of the features were normally distributed, with the exception of 'Distance', 'Departure Delay in Minutes' and 'Arrival Delay in Minutes' which were right-skewed, and 'Age' which was slightly left-skewed and right-skewed. Moreover, we deduced that: 

• **Ratings**: the ratings of the different services provided by the company were mainly positive, with the majority of the customers giving a rating of 4 out of 5; this could be an indicator of a good service provided by the company, but it could also be a sign of a biased dataset, since customers that were not satisfied with the service provided by the company might have decided to not give any rating at all;

• **Distribution of Age**: the age distribution of the customers was quite balanced between 20 and 60 years-old people with a slight skewness towards the younger and older customers; this could be explained by the fact that the company might mainly operate in the business sector (as showned by the majority of customers travveling for work purposes), providing services for business travellers rather than for leisure travellers, which are usually aged between 20 and 60 years old;

• **Distribution of Departure and Arrival Delay in minutes**: the distribution of the departure and arrival delay in minutes was quite similar since these two features are highly correlated, with the majority of the customers experiencing a delay of less than 10 minutes; this could be an indicator of a good service provided by the company, but it could also be a sign of a biased dataset, since again customers that experienced a delay of more than 10 minutes might have decided to not give any rating at all. 
![Alt text](images/num_distribution.png)

