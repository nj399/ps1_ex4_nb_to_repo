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


# Plot count pairs using all_df for the columns: Sex, Pclasss, SibSp, Parch, Embarked
columns_to_plot = ["    Sex", "Pclass", "SibSp", "Parch", "Embarked"]   
for col in columns_to_plot:
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=col, data=all_df, hue="set", palette= color_list)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Number of passengers / {col}")
    plt.show()


# Plot count pairs using all_df for the columns: Sex, Pclasss, SibSp, Parch, Embarked and use "Survived" as hue.
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
columns_to_plot = ["    Sex", "Pclass", "SibSp", "Parch", "Embarked"]   
for col in columns_to_plot:
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=col, data=all_df, hue="Survived", palette= color_list)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y")
    ax.set_title(f"Number of passengers / {col}")
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
    

# Plot distribution pairs for Age and Fare
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
f, ax = plt.subplots(1, 1, figsize=(8, 4))
for i, h in enumerate(train_df["Fare"].unique()):
    g = sns.histplot(train_df.loc[train_df["Age"]==h, "Fare"], 
                                  color=color_list[i], 
                                  ax=ax, 
                                  label=h)
ax.set_title("Number of passengers / Sex")
g.legend()
plt.show()

# Plot distribution pairs for Age and Fare using "Survived" as hue
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
f, ax = plt.subplots(1, 1, figsize=(8, 4))
for i, h in enumerate(train_df["Survived"].unique()):
    g = sns.histplot(train_df.loc[train_df["Survived"]==h, "Fare"], 
                                  color=color_list[i], 
                                  ax=ax, 
                                  label=h)
ax.set_title("Number of passengers / Fare")
g.legend()
plt.show()


all_df["Family Size"] = all_df["SibSp"] + all_df["Parch"] + 1

# Turning this into a function for reuse
def add_family_size(df):
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df       

all_df = add_family_size(all_df)
print(all_df[["SibSp", "Parch", "Family Size"]].head())

train_df["Family Size"] = train_df["SibSp"] + train_df["Parch"] + 1

# function already defined above
train_df = add_family_size(train_df)
print(train_df[["SibSp", "Parch", "Family Size"]].head())
      

# Plot count pairs using all_df for the column "Family Size" and use "Survived" as hue.
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
f, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.countplot(x="Family Size", data=all_df, hue="Survived", palette= color_list)
plt.grid(color="black", linestyle="-.", linewidth=0.5,      axis="y")
ax.set_title("Number of passengers / Family Size")
plt.show()  

# Create Age Intervals
all_df["Age Interval"] = 0.0
all_df.loc[ all_df['Age'] <= 16, 'Age Interval']  = 0
all_df.loc[(all_df['Age'] > 16) & (all_df['Age'] <= 32), 'Age Interval'] = 1
all_df.loc[(all_df['Age'] > 32) & (all_df['Age'] <= 48), 'Age Interval'] = 2
all_df.loc[(all_df['Age'] > 48) & (all_df['Age'] <= 64), 'Age Interval'] = 3
all_df.loc[ all_df['Age'] > 64, 'Age Interval'] = 4


# Turning this into a function for reuse
def add_age_intervals(df):
    df["Age Interval"] = 0.0
    df.loc[ df['Age'] <= 16, 'Age Interval']  = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[ df['Age'] > 64, 'Age Interval'] = 4
    return df       

all_df = add_age_intervals(all_df)
print(all_df[["Age", "Age Interval"]].head())

train_df = add_age_intervals(train_df)
print(train_df[["Age", "Age Interval"]].head())

all_df.head()

# Plot count pairs using all_df for the column "Age Interval" and use "Survived" as hue.
color_list = ["#A5D7E8", "#576CBC", "#19376 D", "#0b2447"]
f, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.countplot(x="Age Interval", data=all_df, hue="Survived",    palette= color_list)
plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y")                    
ax.set_title("Number of passengers / Age Interval")
plt.show()

all_df['Fare Interval'] = 0.0
all_df.loc[ all_df['Fare'] <= 7.91, 'Fare Interval'] = 0
all_df.loc[(all_df['Fare'] > 7.91) & (all_df['Fare'] <= 14.454), 'Fare Interval'] = 1
all_df.loc[(all_df['Fare'] > 14.454) & (all_df['Fare'] <= 31), 'Fare Interval']   = 2
all_df.loc[ all_df['Fare'] > 31, 'Fare Interval'] = 3

def_fare_intervals(df_missing_test):
    df['Fare Interval'] = 0.0
    df.loc[ df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare  <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval']   = 2
    df.loc[ df['Fare'] > 31, 'Fare Interval'] = 3
    return df       

all_df = def_fare_intervals(all_df)
print(all_df[["Fare", "Fare Interval"]].head()) 

train_df = def_fare_intervals(train_df)
print(train_df[["Fare", "Fare Interval"]].head())

# Plot count pairs using all_df for the column "Fare Interval"
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
f, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.countplot(x="Fare Interval", data=all_df, hue="Survived",       
palette= color_list)
plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y")
ax.set_title("Number of passengers / Fare Interval")
plt.show()  


train_df["Sex_Pclass"] = train_df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)

# Turning this into a function for reuse
def add_sex_pclass(df):
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df

train_df = add_sex_pclass(train_df)
print(train_df[["Sex", "Pclass", "Sex_Pclass"]].head())
all_df = add_sex_pclass(all_df)
print(all_df[["Sex", "Pclass", "Sex_Pclass"]].head())   

# Plot count pairs using all_df for the column "Fare Interval" and "Fare (grouped by survival)" with "Survived" as hue
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
f, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.countplot(x="Sex_Pclass", data=all_df, hue="Survived", palette= color_list)
plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y")
ax.set_title("Number of passengers / Sex_Pclass")
plt.show() 

# Parse Names into Family Name, Title, Given Name, Maiden Name
def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")
    

all_df[["Family Name", "Title", "Given Name", "Maiden Name"]] = all_df.apply(lambda row: parse_names(row), axis=1)

# Turning this into a function for reuse
def add_parsed_names(df):
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)
    return df

all_df = add_parsed_names(all_df)
print(all_df[["Name", "Family Name", "Title", "Given Name", "Maiden Name"]].head())

# function already defined above
train_df = add_parsed_names(train_df)
print(train_df[["Name", "Family Name", "Title", "Given Name", "Maiden Name"]].head())


for dataset in [all_df, train_df]:
    dataset["Family Type"] = dataset["Family Size"]

# Turning this into a function for reuse
def add_family_type(df):
    df["Family Type"] = df["Family Size"]
    return df


for dataset in [all_df, train_df]:
    dataset.loc[dataset["Family Size"] == 1, "Family Type"] = "Single"
    dataset.loc[(dataset["Family Size"] > 1) & (dataset["Family Size"] < 5), "Family Type"] = "Small"
    dataset.loc[(dataset["Family Size"] >= 5), "Family Type"] = "Large"

# turning this into a function for reuse
def add_family_type(df):
    df["Family Type"] = df["Family Size"]
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[(df["Family Size"] >= 5), "Family Type"] = "Large"
    return df


for dataset in [all_df, train_df]:
    dataset["Titles"] = dataset["Title"]



for dataset in [all_df, train_df]:
    #unify `Miss`
    dataset['Titles'] = dataset['Titles'].replace('Mlle.', 'Miss.')
    dataset['Titles'] = dataset['Titles'].replace('Ms.', 'Miss.')
    #unify `Mrs`
    dataset['Titles'] = dataset['Titles'].replace('Mme.', 'Mrs.')
    # unify Rare
    dataset['Titles'] = dataset['Titles'].replace(['Lady.', 'the Countess.','Capt.', 'Col.',\
     'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')

# turning this into a function for reuse
def unify_titles(df):
    #unify `Miss`
    df['Titles'] = df['Titles'].replace('Mlle.', 'Miss.')
    df['Titles'] = df['Titles'].replace('Ms.', 'Miss.')
    #unify `Mrs`
    df['Titles'] = df['Titles'].replace('Mme.', 'Mrs.')
    # unify Rare
    df['Titles'] = df['Titles'].replace(['Lady.', 'the Countess.','Capt.', 'Col.',\
     'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
    return df

train_df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()

# turning this into a function for reuse
def title_survival_rate(df):
    return df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()  


for dataset in [train_df, test_df]:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# turning this into a function for reuse
def map_sex(df):
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)   
    return df

# train-validation split
VALID_SIZE = 0.2
train, valid = train_test_split(train_df, test_size=VALID_SIZE, random_state=42, shuffle=True)

# define predictor features and target feature
predictors = ["Sex", "Pclass"]
target = 'Survived'

train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].value

# turning this into a function for reuse
def prepare_train_validation_split(train_df, valid_size=0.2, predictors=[], target='Survived'):
    train, valid = train_test_split(train_df, test_size=valid_size, random_state=42, shuffle=True)
    train_X = train[predictors]
    train_Y = train[target].values
    valid_X = valid[predictors]
    valid_Y = valid[target].values
    return train_X, train_Y, valid_X, valid_Y


# define the model
lf = RandomForestClassifier(n_jobs=-1, 
                             random_state=42,
                             criterion="gini",
                             n_estimators=100,
                             verbose=False)


# fit the model
clf.fit(train_X, train_Y)

# predicte the train data
preds_tr = clf.predict(train_X)

# predict the validation data
preds = clf.predict(valid_X)

# turning this into a function for reuse
def train_evaluate_model(clf, train_X, train_Y, valid_X, valid_Y):
    # fit the model
    clf.fit(train_X, train_Y)

    # predicte the train data
    preds_tr = clf.predict(train_X)

    # predict the validation data
    preds = clf.predict(valid_X)

    return preds_tr, preds


# evaluate the model

print(metrics.classification_report(train_Y, preds_tr, target_names=['Not Survived', 'Survived']))

print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))