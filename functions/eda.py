import pandas as pd
import matplotlib.pyplot as plt


# Creates a csv file with the describe of every column from a dataframe
def describe_data(data):
    describe_df = pd.DataFrame()
    quant = len(data)

    for item in data.columns:
        temp_df = pd.DataFrame(data[item][0:quant].describe()).transpose()
        describe_df = pd.concat([describe_df,temp_df])
    
    missing_values = data.isnull().sum()
    describe_df['missing_values'] = missing_values
    
    return describe_df
    

# Get the distribution and boxplot of large outliers
def distribution_graphs(data, threshold, mode='below'):
    if mode == 'below':
        filtered_data = data[data < threshold]
        title = 'without outliers'
    else:
        filtered_data = data[data > threshold]
        title = 'large outliers'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot boxplot
    ax1.boxplot(filtered_data)
    ax1.set_title(f'Boxplot of distribution {title}')    

    # Plot histogram
    ax2.hist(filtered_data)
    ax2.set_title(f'Histogram of distribution {title}')
    
    plt.show()
 
   
# Calculates the frequency and percentage of unique values in a column
def calculate_unique_values(df, column):
    value_counts = df[column].value_counts()
    percentages = value_counts / len(df) * 100 
    result_df = pd.DataFrame({'Value': value_counts.index, 'Frequency': value_counts.values, 'Percentage': percentages.values})
    
    return result_df