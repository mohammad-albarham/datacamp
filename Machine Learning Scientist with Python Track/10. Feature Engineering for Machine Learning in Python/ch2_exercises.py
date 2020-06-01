# Exercise_1 
#1
# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]

# Print the number of non-missing values
print(sub_df.notnull().sum())
#2
selected_option = 2


--------------------------------------------------
# Exercise_2 
#1
# Print the top 10 entries of the DataFrame
print(sub_df.head(10))
#2
# Print the locations of the missing values
print(sub_df.head(10).isnull())
#3
# Print the locations of the non-missing values
print(sub_df.head(10).notnull())


--------------------------------------------------
# Exercise_3 
#1
# Print the number of rows and columns
print(so_survey_df.shape)
#2
# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna(how='any')

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)
#3
# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how='any', axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)
#4
# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset= ['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)


--------------------------------------------------
# Exercise_4 
#1
# Print the count of occurrences
print(so_survey_df['Gender'].value_counts())
#2
# Replace missing values
so_survey_df['Gender'].fillna('Not Given', inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())


--------------------------------------------------
# Exercise_5 
#1
# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())
#2
# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())
#3
# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())


--------------------------------------------------
# Exercise_6 
#1
# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')
#2
# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$','')


--------------------------------------------------
# Exercise_7 
#1
# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isna()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])
#2
# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£','')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype(float)

# Print the column
print(so_survey_df['RawSalary'])


--------------------------------------------------
# Exercise_8 
# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
                              .str.replace(',','')\
                              .str.replace('$','')\
                              .str.replace('£','')\
                              .astype(float)
 
# Print the RawSalary column
print(so_survey_df['RawSalary'])

--------------------------------------------------
