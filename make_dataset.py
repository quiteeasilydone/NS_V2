import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_a_label(label_path, data_root):
    with open(label_path) as f:
        json_data = json.load(f)
    json_data

    video_name = os.path.join(data_root, json_data['video']['video_name']+'.mp4')
    label = json_data['video']['accident_negligence_rateB']

    data = [video_name, label]

    return data

def extract_a_folder(folder_path, data_root):
    labels = os.listdir(folder_path)
        
    datas = []

    for a_label in labels:
        data = extract_a_label(os.path.join(folder_path,a_label), data_root)
        datas.append(data)
    return datas

def split_data(ratio, dataset_df):
    grouped_class = dataset_df.groupby('label')
    drop_idx = grouped_class.filter(lambda x: len(x) <= 20).index
    dataset_df = dataset_df.drop(drop_idx)
    train_df, val_test_df = train_test_split(dataset_df, test_size = 0.3 , random_state = 7, stratify = dataset_df['label'])
    val_df, test_df = train_test_split(val_test_df, test_size = 0.5 , random_state = 7, stratify = val_test_df['label'])
    
    return train_df, val_df, test_df

def get_ratio_by_class(train_df, val_df, test_df):
    df_distribution = pd.DataFrame()
    df_distribution.insert(0, column='train', value =round(train_df.groupby('label').count()/len(train_df), 2))
    df_distribution.insert(1, column='validation', value =round(val_df.groupby('label').count()/len(val_df), 2))
    df_distribution.insert(2, column='test', value =round(test_df.groupby('label').count()/len(test_df), 2))

    print(df_distribution)


if __name__ =='__main__':

    dataset = extract_a_folder(os.path.join(os.getcwd(),'../data/label'), os.path.join(os.getcwd(),'../data/video'))
    dataset_df = pd.DataFrame(dataset, columns=['paths', 'label'])
    
    train_df, val_df, test_df = split_data(ratio=[0.7, 0.15, 0.15], dataset_df=dataset_df)
    get_ratio_by_class(train_df, val_df, test_df)


    train_path = os.path.join(os.getcwd(), 'train.csv')
    val_path = os.path.join(os.getcwd(), 'val.csv')
    test_path = os.path.join(os.getcwd(), 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)