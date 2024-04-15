import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from joblib import dump
from joblib import load

file1_path = "C:\\Users\\15199\\Downloads\\Capstone Project\\data\\complete.csv"
file2_path = "C:\\Users\\15199\\Downloads\\Capstone Project\\data\\daysviz.csv"

gbmmodel_path = 'gbmclass_model.joblib'
svc_path = 'svc_model.joblib'
XGB_path = 'XGBmodel.joblib'
SMOTE_path = 'SMOTEmodel.joblib'
LGB_path = 'LGBmodel.joblib'
rftuned_path = 'rftunedmodel.joblib'

try:
    df = pd.read_csv(file1_path)
except Exception as e:
    st.error(f"Error reading first CSV file: {e}")

try:
    days = pd.read_csv(file2_path)
except Exception as e:
    st.error(f"Error reading second CSV file: {e}")

df.drop(columns=['Unnamed: 0'], inplace=True)

mag_count = df['mag'].value_counts()

st.markdown("""
    <style>
    .css-1aumxhk {
        max-width: 1200px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Predictive Modeling on Tornado Severity using Weather Data")
st.write("""
This dashboard will walk you through an outline of the processes and major steps that went into my Capstone Project - a project where I combined knowledge from many domains of data science in order to source my own data, create a predictive model from it, then present it along with analysis and visualization.

My original dataset was downloaded straight from: https://www.spc.noaa.gov/ \n
The Storm Prediction Center provides an in depth analysis of all kinds of weather phenomena, however, I downloaded only their CSV on recorded tornadoes in the United States. I chose this dataset not just because of its substantial and well documented records, but also because I wanted the data I was working with to interest me.

The tornado data set was great, but it only included information collected after the tornado had happened - this would make for a poor predictive model, so I combined it with historical weather data, which I sourced with a free, open-source weather API from: https://open-meteo.com/ 

Each tornado recorded included a starting latitude and longitude, and a date and time. Using this API, along with the python library Pandas, I was able to run the API through each line of the tornado data, collecting weather records on the day of the storm, and at the precise location.
""")

fig, ax = plt.subplots(figsize=(8, 6))
mag_count.sort_index().plot(kind='barh', width=0.7, ax=ax)  # Reversing axes
ax.set_ylabel('Magnitude')
ax.set_xlabel('Count')
ax.set_title('Frequencies of Magnitude')
ax.invert_yaxis()
st.pyplot(fig)

st.write("""
The data I used includes tornadoes from 1996 - 2022. The rating system of tornadoes was officially changed in 2007, from the Fujita scale to the Enhanced Fujita scale. However, both these rate tornado intensity from a scale of 0 - 5, the enhanced scale simply reflects closer examinations of damage. As the plot above reflects, a rating of 0 is by far the most common. The highest category of tornadoes are extremely rare.""")


st.write("""
### Tornado Frequency Analysis
Tornadoes happen most frequently in the months of May and June, and throughout the year are most common at 4-7pm, due to the clashing of different temperatures that happen between spring and summer, and afternoon and evening.
""")


month_counts = df['mo'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # Changed to 1 row, 2 columns

# Plot for tornado frequency by month
axes[0].bar(month_counts.index, month_counts.values, color='skyblue')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Tornado Frequency by Month', y=1.02)
axes[0].tick_params(axis='y', which='both', left=False)  # Remove y ticks

# Plot for tornado frequency by hour of the day
axes[1].bar(days['hour'], days['count'], color='darkblue')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Tornado Frequency by Hour of Day', y=1.02)

plt.tight_layout()
st.pyplot(fig)

st.write("""
From the weather data API, I chose 3 weather factors to get hourly data on for each record in my tornadoes dataset. The factors I chose were wind speed at 10m, wind speed at 100m, and atmospheric air pressure reduced to mean sea level. I chose these because tornadoes are created by powerful winds and changes in air pressure. The figure below shows the average wind speeds at both altitudes throughout the day before a tornado occurred, for each magnitude.
""")


def speed_by_mag(magnitude):
    filtered_df = df[df['mag'] == magnitude]

    avg_wind_speed_10m = filtered_df['Average_Wind_Speed_10m'].mean()
    avg_wind_speed_100m = filtered_df['Average_Wind_Speed_100m'].mean()

    positions = [0.5, 0.9]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(positions, [avg_wind_speed_10m, avg_wind_speed_100m], width=0.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(['Wind Speed (10m)', 'Wind Speed (100m)'])
    ax.set_xlabel('Wind Speed Measurement')
    ax.set_ylabel('Average Wind Speed')
    ax.set_title(f'Average Wind Speeds at Different Altitudes for Magnitude {magnitude}')
    ax.set_ylim(0, 40)
    
    return fig, ax

# Add a slider widget for magnitude
magnitude = st.slider("Select Magnitude", 0, 5, 1)

# Call the function with selected magnitude and display plot
fig, ax = speed_by_mag(magnitude)
st.pyplot(fig)


st.write("""
In order to get an equal estimate of the weather leading up to a tornado, I took the average of each factor from the 10 hours immediately preceding a tornado. I wanted a good length of time, which should also be equal for each tornado that I want to predict on. To that end, I dropped all tornadoes that happened before 11am from my dataset. Referring to the frequency analysis above, I feel comfortable dropping these instances since the vast majority of tornadoes occur between 4 and 7pm.
""")

# Modeling and plot
X = df[['Average_Wind_Speed_10m', 'Average_Wind_Speed_100m', 'Average_Pressure_msl']]
y = df['len']

X = sm.add_constant(X)

model = sm.OLS(y,X)
results = model.fit()

# Display model summary
st.header("Selecting a Target")

st.write("""
Now that I have my daily averages leading up to the tornado, I must decide what about the tornado I am going to predict, using these averages. My first thought was to look at the correlation between my weather factors, and the path length of the tornado - how far the tornado traveled (in miles.) To explore this I used an OLS model.
""")

st.text(results.summary().as_text())

st.subheader("Linear Regression Analysis")
st.write("""
The first thing to note here are the coefficients (coef) for each of the 3 weather factors. The significant is measured with the P value (P>|t|). A P value of 0 is good, meaning that the coefficients are significant. The coefficients themselves tell us about the relationship each weather factor has with the target (path length.) Average wind speed at 100m has a coefficient of 0.27. This means for every 1 unit increase in the predictor (wind speed 100m,) the target (path length) increases by 0.27. In other words, the higher the wind speeds at 100m, the further the tornado will travel. The figure below plots the averages against the path length, to give a clear visualization.
""")


# Plotting regression line
coefficient_value = results.params['Average_Wind_Speed_100m']
wind_speed_values = np.linspace(min(df['Average_Wind_Speed_100m']), max(df['Average_Wind_Speed_100m']), 100)
predicted_path_length = coefficient_value * wind_speed_values

plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
plt.scatter(df['Average_Wind_Speed_100m'], df['len'], color='blue', alpha=0.5)
plt.plot(wind_speed_values, predicted_path_length, color='red', label='OLS Coefficient Line')
plt.title('Average Wind Speed at 100m vs Tornado Path Length')
plt.xlabel('Average Wind Speed at 100m in the 10 Hours Leading Up to Tornado')
plt.ylabel('Tornado Path Length')
plt.legend()
plt.grid(True)

# Show the plot
st.pyplot(plt.gcf())

st.write("""
Ultimately, linear regression here is great for explaining the relationships within my dataset, but looking at metrics such as the R-squared in the summary above (0.025) along with others like Mean Squared Error (33.89) and Mean Absolute Error (3.22), it is clear that this model has poor predictive power, so I will go on to explore a different feature.
""")

st.header("Predicting the Category")
st.write("""
Since the model was not able to make accurate predictions on a continuous variable, I decided to go with a classification model instead: using the wind speeds and pressure recorded throughout the day, I want to predict the magnitude of the tornado that follows. Below are some of my first models. The summaries show the overall average, and how well the model does at predicting in each of the categories.
""")

X = df[['Average_Wind_Speed_10m', 'Average_Wind_Speed_100m', 'Average_Pressure_msl']]
y = df['mag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("Gradient Boosted Model")

gbmclass = load('gbmclass_model.joblib')
y_pred_gbm = gbmclass.predict(X_test)

accuracy_gbm = accuracy_score(y_test, y_pred_gbm)
st.write(f"Accuracy: {accuracy_gbm:.2f}")

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred_gbm))

st.subheader("Support Vector Class")

svmclass = load('svc_model.joblib')

y_pred_svm = svmclass.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
st.write(f"SVM Accuracy: {accuracy_svm:.2f}")

st.write("SVM Classification Report:")
st.text(classification_report(y_test, y_pred_svm))

#MAG CHANGES

df['mag'] = df['mag'].replace(5, 4)
df['mag'] = df['mag'].replace(4, 3)
df['mag'] = df['mag'].replace(3, 2)

st.header("Difficulties & Solution")
st.write("""
These two models, GBM and SVC are among the first I tried, along with other common classifiers such as logistic regression, to get an idea of how my model was attempting to categorize the weather data. Looking at the classification report and accuracy scores, we can see that while the overall accuracy is above 60%, which is reasonable at this time, it's only scoring that high because it predicts into the most obvious categories, and disregards the smaller ones. My solution to this was to combine the higher magnitude ratings into one 'High Severity' category. The new categories describe 3 levels of tornado severity:
""")

st.subheader("Low Severity")
st.write("""
This category includes F0 and EF0 tornadoes. As we saw in our counts of magnitude from earlier, roughly 50% of all tornadoes fall into this classification. Tornadoes of this intensity cause minor damage. Small trees are blown down and bushes uprooted. Shingles are ripped off roofs, windows in cars and buildings can be blown out, loose and small items are tossed and blown away.
""")

st.subheader("Medium Severity")
st.write("""
This contains F1 and EF1 tornadoes, they make up roughly 30% of all tornadoes, and cause moderate damage. Roofs are stripped of shingles, and small areas may be blown off houses. Doors are blown in, siding ripped off houses, small trees uprooted, large trees snapped or blown down, cars occasionally flipped or blown over.
""")

st.subheader("High Severity")
st.write("""
This category contains the remaining 20% of tornadoes, from F/EF2 through F/EF5. Considerable - devastating damage. At the low end, whole roofs are ripped off frame houses, large trees uprooted, weak structures such as barns and mobile homes are completely destroyed. At the high end, nearly all buildings aside from heavily built structures are destroyed, cars are mangled and thrown hundred of meters away, wood and any small solid material becomes dangerous projectiles
""")

X = df[['Average_Wind_Speed_10m', 'Average_Wind_Speed_100m', 'Average_Pressure_msl']]
y = df['mag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


st.header("Hyperparameter Tuning and Model Exploration")

st.write("""
Despite making the categories more even, my models could still be improved. It was at this point that I began testing different hyperparameters - numerous and specific parameters that describe how the model learns. Adding new ones, and testing different combination for each of them can be an arduous task, which is why I selected the most promising models, and tested many different hyperparameters at once with grid search and randomized cross validation, a process which automatically tests various hyperparameters and their values to find the optimal settings.
""")

st.subheader("XGB")
st.write("XGBoost was a great model, and with some hyper parameter tuning I was able to keep the accuracy at 60, while still predicting into each of my classes. The amount of predictions into the medium and high severity categories were just a little too low though, so I continued to look at other models to see if that could be improved.")

XGBclass = load('XGBmodel.joblib')

y_pred_xgb = XGBclass.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
st.write(f"XGBM Accuracy: {accuracy_xgb:.2f}")

st.write("XGBM Classification Report:")
st.text(classification_report(y_test, y_pred_xgb))


st.subheader("SMOTE")
st.write("SMOTE uses oversampling, randomly generating new instances in smaller classes. This was great for spreading the predictions out, as we see with the scores in each category, however the accuracy suffered greatly.")

SMOTEclass = load('SMOTEmodel.joblib')

y_pred_smote = SMOTEclass.predict(X_test)

accuracy_smote = accuracy_score(y_test, y_pred_smote)
st.write(f"SMOTE Accuracy: {accuracy_smote:.2f}")

st.write("SMOTE Classification Report:")
st.text(classification_report(y_test, y_pred_smote))


st.subheader("Random Forest Tuned")
st.write("Random Forest was one of the first models I tried, and it even managed to predict some smaller categories when there were still 5. For that reason, I included it in my hyperparameter testing, and using cross validation, along with some experimenting of my own, I was able to find a set of hyperparameters which also predicted the category with 60%, but had a little bit more support for the smaller classes than XGBoost did.")

rftunedclass = load('rftunedmodel.joblib')

y_pred_rftuned = rftunedclass.predict(X_test)

accuracy_rftuned = accuracy_score(y_test, y_pred_rftuned)
st.write(f"Random Forest Tuned Accuracy: {accuracy_rftuned:.2f}")

st.write("Random Forest Tuned Classification Report:")
st.text(classification_report(y_test, y_pred_rftuned))

st.header("Final thoughts")
st.write("Out of the models I tried, this tuned random forest gave me the best results - balancing accuracy without abandoning the small classes. One hyperparameter in particular that helped here was class weights. In theory, this is similar to what SMOTE attempted to do, telling my model to focus more on these specific classes, despite their small size. Along with lowering the learning rate, I am satisfied with this random forest model. In conclusion: with this model, when provided with the average wind speeds at 10m and 100m, along with the average pressure (msl) on any given day, if a tornado begins to form, we can predict how severe that tornado will be, nearly two thirds of the time!")

