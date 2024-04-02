import pandas as pd
# pip install mlxtend 
# TransactionEncoderused for converting the transaction data into a one-hot encoded DataFrame. 
# Valid form that is used by Apriori algorithm as its parameter.
from mlxtend.preprocessing import TransactionEncoder 
# apriori is the Apriori algorithm that we will use to mine the frequent itemsets.
from mlxtend.frequent_patterns import apriori
# association_rules is the function that we will use to extract the rules from the frequent itemsets.
from mlxtend.frequent_patterns import association_rules

# Read the datasets
df4 = pd.read_csv('FilteredSport4.csv')
df5 = pd.read_csv('FilteredSport5.csv')

# Concatenate the datasets
combined_df = pd.concat([df4, df5], ignore_index=True)
# print(combined_df)

# Fill missing values, if any
combined_df = combined_df.fillna('')

# convert the 'id' column to str
combined_df['ime_aktivnosti'] = combined_df['ime_aktivnosti'].astype(str)
# print(combined_df['ime_aktivnosti'])

# Convert the DataFrame to a list of lists (transactions)
transactions = combined_df.groupby('id')['ime_aktivnosti'].apply(list).tolist()
# print(transactions)


# Use TransactionEncoder to convert the transaction data into a one-hot encoded DataFrame
te = TransactionEncoder() # Create an instance of TransactionEncoder
te_ary = te.fit(transactions).transform(transactions) # Use fit and transform to one-hot encode the transaction data
df_encoded = pd.DataFrame(te_ary, columns=te.columns_) # Create a DataFrame from the one-hot encoded data
# print(te_ary)
# print(transactions)
# print(df_encoded)

# Apply the Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True) 
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)

# Display the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)
# Display the rules
print("\nMined Associative Rules:")
print(rules)
# In simpler terms, these rules are suggesting relationships between different types of activities ("cycling" and "biking") based on how often they occur together in your dataset. For example, the first rule suggests that if someone is engaged in "cycling," there is a high likelihood (confidence) that they will also be engaged in "biking." 