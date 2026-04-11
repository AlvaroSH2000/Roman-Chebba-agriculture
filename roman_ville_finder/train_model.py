import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
plt.style.use('seaborn-v0_8-whitegrid')

from utils import mapping

train_data = pd.read_excel("../data/processed/other_sites_processed.xlsx", sheet_name="Sheet1")
train_data = train_data.drop(columns=["FID"])
# 75% of the data is selected
train_df = train_data.sample(frac=0.75, random_state=4)
# it drops the training data
# from the original dataframe
test_df = train_data.drop(train_df.index)

# mapping the data to the range of 0 and 1
Type_max = train_df["Type"].max()
Type_min = train_df["Type"].min()
Road_max = train_df["Road_dist"].max()
Road_min = train_df["Road_dist"].min()
Near_min = train_df["Near_3000"].min()
Near_max = train_df["Near_3000"].max()
Landform_max = train_df["Landform_agri_score"].max()
Landform_min = train_df["Landform_agri_score"].min()
Soil_max = train_df["Soil_agri_score"].max()
Soil_min = train_df["Soil_agri_score"].min()
Aquifer_max = train_df["Aquifer_productivity"].max()
Aquifer_min = train_df["Aquifer_productivity"].min()

train_df["Road_dist"] = mapping(train_df["Road_dist"], Road_min, Road_max)
train_df["Near_3000"] = mapping(train_df["Near_3000"], Near_min, Near_max)
train_df["Landform_agri_score"] = mapping(train_df["Landform_agri_score"], Landform_min, Landform_max)
train_df["Soil_agri_score"] = mapping(train_df["Soil_agri_score"], Soil_min, Soil_max)
train_df["Aquifer_productivity"] = mapping(train_df["Aquifer_productivity"], Aquifer_min, Aquifer_max)
train_df["Type"] = mapping(train_df["Type"], Type_min, Type_max)

test_df["Road_dist"] = mapping(test_df["Road_dist"], Road_min, Road_max)
test_df["Near_3000"] = mapping(test_df["Near_3000"], Near_min, Near_max)
test_df["Landform_agri_score"] = mapping(test_df["Landform_agri_score"], Landform_min, Landform_max)
test_df["Soil_agri_score"] = mapping(test_df["Soil_agri_score"], Soil_min, Soil_max)
test_df["Aquifer_productivity"] = mapping(test_df["Aquifer_productivity"], Aquifer_min, Aquifer_max)
test_df["Type"] = mapping(test_df["Type"], Type_min, Type_max)

# split the data into features and labels
X_train = train_df.drop("Type", axis=1)
X_test = test_df.drop("Type", axis=1)
y_train = train_df["Type"]
y_test = test_df["Type"]

input_shape = [X_train.shape[1]]

# Creating a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, activation="relu", input_shape=input_shape),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(1)
])
model.summary()
model.compile(optimizer="adam", loss="mae")

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000)

# Save the model
# model.save('./model/trained_model.keras')
# mapping_info = pd.DataFrame({"wt_min": [wt_min], "wt_max": [wt_max], "nt_min": [nt_min], "nt_max": [nt_max], "result_min": [result_min], "result_max": [result_max]})
# mapping_info.to_csv('./model/mapping_info.csv', index=False)
# print("Model saved to ./model/trained_model.keras")

history_df = pd.DataFrame(history.history)
history_df.plot()
plt.show()

