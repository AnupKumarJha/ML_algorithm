#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for easier visualization
import seaborn as sns
# import color maps
from matplotlib.colors import ListedColormap

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")




#for printing floating point numbers upto  precision 2
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:.2f}'.format

# Load real estate data from CSV or Excel file
print("Loading ........... data from CSV or Excel file")
df=pd.read_csv("Churn_Modelling.csv")
print("----------------------------------------------------------------------------")


# Display the dimensions of the dataset.
print("Displaying the dimensions of the dataset--")
print("df.shape=",df.shape)
print("----------------------------------------------------------------------------")
print("RENAME SPECIFIC COLUMNS")


# Columns of the dataset
print("Columns of the dataset--")
print("Columns of the dataset",df.columns)
print("----------------------------------------------------------------------------")




# Display the first 5 rows to see example observations.
print("Display the first 5 rows to see example observations--")
pd.set_option('display.max_columns', 20) ## display max 20 columns
print("data head=")
print(df.head())
print("----------------------------------------------------------------------------")


# Some feaures are numeric and some are categorical
# Filtering the categorical features:
print("Some feaures are numeric and some are categorical Filtering the categorical features:")
print(df.dtypes[df.dtypes=='object'])


# Distributions of numeric features
# One of the most enlightening data exploration tasks is plotting the distributions of your features.

# Plot histogram grid
print("Plotting histogram grid")
# df.hist(figsize=(16,16), xrot=-45) ## Display the labels rotated by 45 degress

# Clear the text "residue"
# plt.show()
print("----------------------------------------------------------------------------")

# Display summary statistics for the numerical features.
print("Display summary statistics for the numerical features")
print("data describe()")
print(df.describe())
print("----------------------------------------------------------------------------")


# Distributions of categorical features
# Display summary statistics for categorical features
print("Distributions of categorical features,Display summary statistics for categorical features")
print(" describe the categorical feature=")
print(df.describe(include=['object']))
print("----------------------------------------------------------------------------")


# Bar plots for categorical Features
# Plot bar plot for the 'exterior_walls' feature.
print("Bar plots for categorical Features Plot bar plot for the 'exterior_walls' feature.")
# plt.figure(figsize=(8,8))
# sns.countplot(y='Geography', data=df)
# plt.show()
# sns.countplot(y='Gender', data=df)
# plt.show()
# do this for multiple feature
print("----------------------------------------------------------------------------")


print("taking care of sparse class if the categorical feature is present")
print("Sparse classes are classes in categorical features that have a very small number of observations.")
print("They tend to be problematic when we get to building models.")
print("In the best case, they don't influence the model much.In the worst case, they can cause the model to be overfit.")
print("Let's make a mental note to combine or reassign some of these classes later using drawn bar plot")
# code here to remove the
print("----------------------------------------------------------------------------")



print("Segmentations")
print("Segmentations are powerful ways to cut the data to observe the relationship between categorical features and numeric features.")
print("Segmenting the target variable by key categorical features.")
# sns.boxplot(y='Geography', x='Exited', data=df)
# plt.show()
# sns.boxplot(y='Gender', x='Exited', data=df)
# plt.show()
print("----------------------------------------------------------------------------")


print("Let's compare the any catorical feateare across other features as well")
print("displaying the means and standard deviations within each class")
print(df.groupby('Geography').mean())
print("----------------------------------------------------------------------------")



print("Correlations")
print("Positive correlation means that as one feature increases, the other increases; eg. a child's age and her height.")
print("Negative correlation means that as one feature increases, the other decreases;")
print("Correlations near -1 or 1 indicate a strong relationship.")
print("Those closer to 0 indicate a weak relationship.")
print("0 indicates no relationship.")

print("correlation=",df.corr())
print("----------------------------------------------------------------------------")

print("A lot of numbers make things difficult to read. So let's visualize this.")
# plt.figure(figsize=(30,30))
# sns.heatmap(df.corr())
# plt.show()
print("Dark colors indicate strong negative correlations and light colors indicate strong positive correlations")
print("he most helpful way to interpret this correlation heatmap is to first find features that are correlated with our target variable by scanning the first column")
print("----------------------------------------------------------------------------")

print("Plotting the more elobrative way")
# mask=np.zeros_like(df.corr())
# mask[np.triu_indices_from(mask)] = True
# plt.figure(figsize=(10,10))
# with sns.axes_style("white"):
#     ax = sns.heatmap(df.corr()*100, mask=mask, fmt='.0f', annot=True, lw=1, cmap=ListedColormap(['green', 'yellow', 'red','blue']))
# plt.show()
print("----------------------------------------------------------------------------")

print("##########################################################################################")
print("Untill now we have anylased the data Now we will begin to Clean the data")
print("##########################################################################################")

print("Dropping the duplicates (De-duplication)")
df = df.drop_duplicates()
print("is any shape change=",df.shape )
print("*****************************************************************************")

print("Fix structural errors")
print("First find the if any nan value is there")
# print(df."".unique())
print("*****************************************************************************")


print("fixingn NaN values ")
# df.column Name.fillna("here value to inserted to replace", inplace=True)
print("cheching for confirmation")
# print(df."Column name".unique())
print("*****************************************************************************")


# Typos and capitalization
print("Typos and capitalization")
# sns.countplot(y='any column name prefarabley categorical value', data=df)
print("composition' should be 'Composition'")
print("renaming if any typos is there")
# df.COLUMN NAME.replace('composition', 'Composition', inplace=True)
print("again plotting")
# sns.countplot(y='COLUMN NAME', data=df)
print("*****************************************************************************")


print("Mislabeled classes")
print("Finally, we'll check for classes that are labeled as separate classes when they should really be the same")
print("e.g. If 'N/A' and 'Not Applicable' appear as two separate classes, we should combine them")
print("let's plot the class distributions for 'ANY COLUMN NAME'")
# sns.countplot(y='exterior_walls', data=df)
# df.exterior_walls.replace(['Rock, Stone'], 'Masonry', inplace=True)
# df.exterior_walls.replace(['Concrete', 'Block'], 'Concrete Block', inplace=True)
# sns.countplot(y='exterior_walls', data=df)
print("*****************************************************************************")


print("Removing Outliers------->")
print("Outliers can cause problems with certain types of models.")
print("Boxplots are a nice way to detect outliers")
print("Let's start with a box plot of your target variable, since that's what you're actually trying to predict.")

# sns.boxplot(df.Metro_dist)
# sns.boxplot(df.no_of_store)
# sns.boxplot(df.latitude)
# sns.boxplot(df.longitude)
# sns.boxplot(df.price)
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(df['Metro_dist'], df['price'])
# ax.set_xlabel('Proportion of non-retail business acres per town')
# ax.set_ylabel('Full-value property-tax rate per $10,000')
# plt.show()
print("*****************************************************************************")


# Label missing categorical data
print("Label missing categorical data")
print("You cannot simply ignore missing values in your dataset. You must handle them in some way for the very practical reason that Scikit-Learn algorithms do not accept missing values")
# Display number of missing values by categorical feature

print(df.select_dtypes(include=['object']).isnull().sum())
# The best way to handle missing data for categorical features is to simply label them as 'Missing'¶
print("he best way to handle missing data for categorical features is to simply label them as 'Missing'")

# df['exterior_walls'] = df['exterior_walls'].fillna('Missing')
# df['roof'] = df['roof'].fillna('Missing')
# df.select_dtypes(include=['object']).isnull().sum()
# Flag and fill missing numeric data
print("Flag and fill missing numeric data")
print(df.select_dtypes(exclude=['object']).isnull().sum())
print("*****************************************************************************")

print("Before we move on to the next module, let's save the new dataframe we worked hard to clean.")
# Save cleaned dataframe to new file
df.to_csv('cleaned_Churn_modelling.csv', index=None)

#Feature Engineering
print("Feature Engineering")
print("For example, let's say you knew that homes with 2 bedrooms and 2 bathrooms are especially popular for investors.")

# Display percent of rows where two_and_two == 1
# df[df['two_and_two']==1].shape[0]/df.shape[0]

# df['cheap']=(df.price<100).astype(int)
# #display the percentage of properties the are cheap
# # df[df['cheap']==1].shape[0]/df.shape[0]
# print("shape=",df['cheap'])
# print("Columns of the dataset--")
# print("Columns of the dataset",df.columns)
# print(df.head(5))
print("*****************************************************************************")

# dropping any column
print("dropping any column")
#dropping the basement column
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)
# df = df.drop(['cheap'], axis=1)
print(df.columns)
print("*****************************************************************************")

# Handling Sparse Classes
print("he easiest way to check for sparse classes is simply by plotting the distributions of your categorical features")
# Bar plot for exterior_walls
# sns.countplot(y='Column Name', data=df)
# For example Group 'Wood Siding', 'Wood Shingle', and 'Wood' together. Label all of them as 'Wood'.
# df.exterior_walls.replace(['Wood Siding', 'Wood Shingle', 'Wood'], 'Wood', inplace=True)
print("*****************************************************************************")


# Encode dummy variables (One Hot Encoding)
print("Encode dummy variables (One Hot Encoding) if any")
df = pd.get_dummies(df, columns=['Gender', 'Geography'])
print("*****************************************************************************")



# Remove unused or redundant features
print("Remove unused or redundant features if any")
print("*****************************************************************************")


print("lets save this model")
print("inally, before we move on to the next module, let's save our new DataFrame we that augmented through feature engineering.")
print(" We'll call it the analytical base table because we'll be building our models on it.")
print("Remember to set the argument index=None to save only the data.")
df.to_csv('analytical_Churn_Modelling.csv', index=None)

print("#########################################Machine Learning Models#################################################")
df = pd.read_csv("analytical_Churn_Modelling.csv")
print("shape=",df.shape)

#dividing the
X = df.iloc[:, 0:11].values
y = df.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages

from keras.models import  Sequential
from keras.layers import Dense

#Initialising the Ann
clasifier=Sequential()

#adding the input layer
clasifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#Adding the secong headen layer
clasifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#adding output layer
# clasifier.add((Dense(outout_dim=1,init='uniform',activation='sigmoid')))
clasifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#Compiling the ANN
clasifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ANN to training set
clasifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)
# Predicting the Test set results
y_pred = clasifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
