import pandas as pd
import urllib.request
import os
import numpy as np
import argparse
from itertools import repeat
from multiprocessing.dummy import Pool as ThreadPool

def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def get_img(url, save_path):
    failed_count = 0
    success_count = 0
    try:
        urllib.request.urlretrieve(url, save_path)
        success_count += 1
    except:
        # print('failed {}'.format(save_path))
        # print('img url {}'.format(url))
        failed_count += 1

    return [success_count, failed_count]

def get_data(csv_filepath, save_path, num_thread, start_idx=0):
    img_pd = preprocess_csv(csv_filepath)
    len_img = len(img_pd)
    len_usr = len(set(img_pd["profile_id"]))
    print('Start downloading {} images of {} users at index {}'.format(len_img, len_usr, start_idx))

    data_folder = save_path

    if os.path.exists(data_folder) is False:
        os.mkdir(data_folder)

    failed_count = 0
    success_count = 0
    user_mobile_list = img_pd['profile_id'].tolist()
    
#     start_idx = 0

    for i in range(start_idx, len_img, num_thread):
        img_save_list = []
        url_list = []
        if i+num_thread >= len_img:
            tmp_pd = img_pd.iloc[i:len_img-1, :]
            for idx, row in tmp_pd.iterrows():
                img_save_name = '_'.join([str(row['profile_id']), str(row['photo_id']), 
                                      str(row['upto_date']), str(row['created_date']),
                                      str(row['birthyear']), str(row['gender'])]) + '.jpg'
                img_save_list.append(os.path.join(data_folder, img_save_name))
                url_list.append(row['source'])
        else:
            tmp_pd = img_pd.iloc[i:i+num_thread, :]
            for idx, row in tmp_pd.iterrows():
                img_save_name = '_'.join([str(row['profile_id']), str(row['photo_id']), 
                                      str(row['upto_date']), str(row['created_date']),
                                      str(row['birthyear']), str(row['gender'])]) + '.jpg'
                img_save_list.append(os.path.join(data_folder, img_save_name))
                url_list.append(row['source'])
        
        pool = ThreadPool(num_thread)
        result = np.array(pool.starmap(get_img, zip(url_list, img_save_list)))
        pool.close()
        pool.join()

        result = np.sum(result,axis=0)
        failed_count += result[1]
        success_count += result[0]
        
        if i % (100*num_thread) == 0:
            print('progress {:.5f}%'.format(100*(i+num_thread)/len_img))
            print('currently at idx {}'.format(i+num_thread))
            print('success count {}'.format(success_count))
            print('failed count {}\n\n'.format(failed_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to csv file contain img source", required=True)
    parser.add_argument("--output", help="path to output images folder", required=True)
    parser.add_argument("--num-thread", help="number of threads for joblib", required=True, type=int)
    parser.add_argument("--start-index", help="index to start downloading", required=False, default=0, type=int)
    args = parser.parse_args()
    get_data(args.input, args.output, args.num_thread, args.start_index)