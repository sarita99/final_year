##this code will help you combine all the csv files having a similar pattern of names containing tweets 

import os
import glob
import pandas as pd

path = "C:/Users/KISHOR/.spyder-py3/codes/"

#this gets all the files that have the same pattern in the name
all_files = glob.glob(os.path.join(path,"covid19_tweets_*.csv"))

#merge all the files with tweets from all the dates

df_merged = (pd.read_csv(f, sep=',') for f in all_files)

df_merged = pd.concat(df_merged)
data_new = df_merged.drop_duplicates()

#dx = data_new.drop(['created_at','user_location'],axis=1)

data_new.to_csv("covid19_tweets_02_date.csv",index=False)


###########3
# df = pd.read_csv("covid19_tweets_14.csv")

# df.to_csv("covid19_tweets.csv",mode = 'a',index=False, encoding="utf-8")

# ###############3
# df = pd.read_csv('covid19_tweets_12_1.csv')
# df1 = pd.read_csv('covid19_tweets_12.csv')

# dfx = df1.drop(['created_at','user_location'],axis=1)
# dfx=dfx[0:226]
# print(df.equals(dfx))
# ###############################
# df= pd.read_csv('covid19_tweets_02_date.csv')
# df1 = pd.read_csv('covid19_tweets_22.csv')

# result = pd.concat([df,df1],axis=0,ignore_index=True)

# result.to_csv('covid19_tweets_02_date.csv',columns=['created_at','original_text','user_location'],index=False, encoding="utf-8")
