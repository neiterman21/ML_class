import pandas as pd
import readData
def p1_distance(input_data, training_set):
    distance_diff = training_set.sub( input_data,  axis='columns')
    distance_diff = distance_diff.abs()
    return distance_diff.sum(axis=1)

def p2_distance(input_data, training_set):
    distance_diff = training_set.sub( input_data,  axis='columns')
    distance_diff = distance_diff**2
    return distance_diff.sum(axis=1)**0.5

def p2_inf_distance(input_data, training_set):
    distance_diff = training_set.sub( input_data,  axis='columns')
    distance_diff = distance_diff.abs()
    return distance_diff.max(axis=1)

def classify(input_data, training_set, labels, distance_func ,k=1):

    distance = distance_func(input_data,training_set)
    distance_df = pd.concat([distance, labels], axis=1)
    distance_df.sort_values(by=[0], inplace=True)
    top_knn = distance_df[:k]
    return top_knn['label'].value_counts().index.values[0]

if __name__ == "__main__":
    df , labels = readData.hbt_data('HC_Body_Temperature.txt')
    print(classify(pd.Series(df.tail(1).reset_index(drop=True).iloc[0]),df.head(65),labels.head(65),p2_inf_distance,k=5) )
