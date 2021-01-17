import pandas as pd


def hbt_data(file):
    hbt = pd.DataFrame(data=pd.read_csv(file, delimiter=r"\s+", engine='python', names=['c1','label','c2']))
    hbt = hbt[hbt.columns[[0,2,1]]]
    hbt.loc[hbt['label'] == 2, 'label'] = -1
    hbt = hbt.sample(frac=1).reset_index(drop=True)
    labels = hbt['label']
    hbt.drop('label' , axis='columns',inplace=True)
    return hbt , labels

if __name__ == "__main__":
    print(hbt_data('HC_Body_Temperature.txt').shape)
