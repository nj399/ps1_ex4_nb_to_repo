import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Hard coded parameters
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

# Load Data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Initial Data Exploration
train_df.head(10)
test_df.head(10)
train_df.info()
test_df.info()
train_df.describe()
test_df.describe()

# Missing Values in Train Data
total = train_df.isnull().sum()
percent = (train_df.isnull().sum()/train_df.isnull().count()*100)
tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
types = []
for col in train_df.columns:
    dtype = str(train_df[col].dtype)
    types.append(dtype)
tt['Types'] = types
df_missing_train = np.transpose(tt)

def missing_values_table(df):
    # Total missing values
    missing_value = df.isnull().sum()

    # Percentage of missing values
    ## isnull checks for NaN values, len(df) gives total number of rows
    missing_value_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    ## concat = joins two dataframes together
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1, keys = ['Missing Values', '% of Total Values'])

    # Transpose
    df_missing = tt.transpose()

    # Return the dataframe with missing information
    return df_missing

df_missing_train = missing_values_table(train_df)

# Note: since we've already defined df_missing_train above, we can reuse the function to check missing values in test data
df_missing_test = missing_values_table(test_df)

print(df_missing_train)

print(df_missing_test)

# Most Frequent Items in Train Data
total = train_df.count()
tt = pd.DataFrame(total)
tt.columns = ['Total']
items = []
vals = []
for col in train_df.columns:
    try:
        itm = train_df[col].value_counts().index[0]
        val = train_df[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    except Exception as ex:
        print(ex)
        items.append(0)
        vals.append(0)
        continue
tt['Most frequent item'] = items
tt['Frequence'] = vals
tt['Percent from total'] = np.round(vals / total * 100, 3)
np.transpose(tt)

# Turning this into a function for reuse
def most_frequent_items_table(df):
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
        continue

    tt['Most frequent item'] = items
    tt['Frequency'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    np.transpose(tt)
    return np.transpose(tt)

df_most_frequent_train = most_frequent_items_table(train_df)
df_most_frequent_test = most_frequent_items_table(test_df)

print(df_most_frequent_train)
print(df_most_frequent_test)

# Unique Values in Train Data
total = train_df.count()
tt = pd.DataFrame(total)
tt.columns = ['Total']
uniques = []
for col in train_df.columns:
    unique = train_df[col].nunique()
    uniques.append(unique)
tt['Uniques'] = uniques
np.transpose(tt)

# Turning this into a function for reuse
def unique_values_table(df):
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    np.transpose(tt)
    return np.transpose(tt)

df_unique_train = unique_values_table(train_df)
df_unique_test = unique_values_table(test_df)
print(df_unique_train)

# function already defined above
print(df_unique_test)

# Combine Train and Test Data for Further Analysis
all_df = pd.concat([train_df, test_df], axis=0)
all_df["set"] = "train"
all_df.loc[all_df.Survived.isna(), "set"] = "test"

# turning this into a function for reuse
def combine_train_test(train_df, test_df):
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df

all_df = combine_train_test(train_df, test_df)
print(all_df.head())

color_list = ["#66b3ff", "#99ff99", "#ff9999", "#ffcc99", "#c2c2f0", "#ffb3e6"]

# Plot count pairs "Sex"
f, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.countplot(x="Sex", data=all_df, hue="set", palette= color_list)
plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
ax.set_title("Number of passengers / Sex")
plt.show()  

# Plot distribution pairs for "Sex" and hue as "Survived"
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
f, ax = plt.subplots(1, 1, figsize=(8, 4))
for i, h in enumerate(train_df["Survived"].unique()):
    g = sns.histplot(train_df.loc[train_df["Survived"]==h, "Sex"], 
                                  color=color_list[i], 
                                  ax=ax, 
                                  label=h)
ax.set_title("Number of passengers / Sex")
g.legend()
plt.show()



