import pandas as pd
import os
import io



def convert_to_csv(path,out_file):
    pos_path = path + 'pos/'
    neg_path = path + 'neg/'
    print('Reading Files..')
    text = [io.open(pos_path+file_).read() for file_ in os.listdir(pos_path)] +\
             [io.open(neg_path+file_).read() for file_ in os.listdir(neg_path)]
    label= ['Positive' for _ in os.listdir(pos_path) ] +\
            ['Negative' for _ in os.listdir(neg_path) ]

    df = pd.DataFrame()
    df['text'] = text
    df['sentiment']= label
    print('Writing Data in CSV..')
    df.to_csv(out_file,encoding='utf8',index=False)
    print('Processing Complete.')

if __name__ == '__main__':
    TRAIN_PATH='aclImdb/train/'
    TEST_PATH='aclImdb/test/'
    TRAIN_CSV_FILE='train.csv'
    TEST_CSV_FILE='test.csv'
    convert_to_csv(TRAIN_PATH,TRAIN_CSV_FILE)
    convert_to_csv(TEST_PATH,TEST_CSV_FILE)
