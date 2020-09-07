##this code is to topic modeling the data(tweets).Tweets are preprocessed (cleaned) and the algorithms are applied
##total number of tweets pulled = 2972

import nltk
#from nltk.tokenize import RegexpTokenizer
#from nltk.corpus import stopwords
import pandas as pd
import re
import matplotlib.pyplot as plt
#import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,NMF

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # remove bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet, @user and hashtags information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)#emove hastags
    return tweet

my_stopwords = nltk.corpus.stopwords.words('english')
#word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

# cleaning master function
def clean_tweet(tweet, bigrams=False):
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)#remove non-ASCII characters
    tweet = re.sub('&amp','',tweet)#removes the string &amp
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = re.sub('#','',tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('via namo app','',tweet)#remove the substring = "via namo app"
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = tweet.strip()#removes whitespaces from before and after strings
    
    tweet_token_list = [word for word in tweet.split(' ')
                            if word not in my_stopwords] # remove stopwords

    #tweet_token_list = [word_rooter(word) if '#' not in word else word
                        #for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)   

#read the csv file containing the original tweets
df = pd.read_csv("covid19_tweets_02_22.csv")
df = df.drop(['created_at','user_location'],axis=1)
df = df.rename(columns = {'original_text':'tweet'})#rename the column

#removing "via namo app" substring from the tweets to improve results

# make new columns for retweeted usernames, mentioned usernames and hashtags
df['mentioned'] = df.tweet.apply(find_mentioned)
df['hashtags'] = df.tweet.apply(find_hashtags)
df['clean_tweet'] = df.tweet.apply(clean_tweet)
##df.shape[0] = 2972

#its observed that some tweets after cleaning are empty, hence lets remove them
df.drop(df[df['clean_tweet'].map(len)==0].index, inplace = True)
#df.shape[0] = 2957
df = df.reset_index(drop=True)
#df = df.sample(frac=1).reset_index(drop=True)#shuffle data

df["Tweet length"] = df["clean_tweet"].str.len()

x =[]
for i in range(df.shape[0]):
    if (df["Tweet length"][i].item()<=10):
        x.append(i)#130 items

#dropping tweets with length <=10        
df.drop(df[df['clean_tweet'].map(len)<=10].index, inplace = True)#len 2827
df = df.reset_index(drop=True)

df.to_csv("final_tweets_02_date.csv",index=False)

########################
def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

no_top_words = 10
#######################################Applying LDA model################################################

# the vectorizer object will be used to transform text to vector form
tf_vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# apply transformation(convert it to string first)
tf = tf_vectorizer.fit_transform(df['clean_tweet'].values.astype('U'))

# tf_feature_names tells us what word each column in the matrix represents
tf_feature_names = tf_vectorizer.get_feature_names()

number_of_topics = 10

model1 = LatentDirichletAllocation(n_components=number_of_topics,random_state=0)
model1.fit(tf)

table1 = display_topics(model1, tf_feature_names, no_top_words)
table1.to_csv('results_lda3.csv',index=False)

#gives the document term matrix
doc_topic = model1.transform(tf)
lda_doc_topic = pd.DataFrame(doc_topic)
lda_doc_topic.to_csv('lda_doctop3.csv',index = False)

doc = []
topic = []
tweets = []
for n in range(lda_doc_topic.shape[0]):
    doc.append(n)
    topic_most_pr = doc_topic[n].argmax()
    topic.append(topic_most_pr)
    tweets.append(df.iloc[n,3])
    #print("doc: {} topic: {}\n".format(n,topic_most_pr))
    
dfdoc = pd.DataFrame()
dfdoc["Topic"] = topic
dfdoc["Topic prob"] = lda_doc_topic.max(axis=1)

maxes = lda_doc_topic.max(axis=1)
less_than_max = lda_doc_topic.where(lda_doc_topic.lt(maxes, axis='rows'))
seconds = less_than_max.max(axis=1)
dfdoc['diff'] = maxes - seconds

dfdoc["Document"] = doc
dfdoc["Tweet"] = tweets
dfdoc = dfdoc.sort_values(by=["Document"])
dfdoc.to_csv("lda_doc3.csv",index=False)

#sum(round(dfdoc["Topic prob"],1)<0.5) = 952(should decrease)
#sum(round(dfdoc["diff"],1)>=0.5) = 1187(should increase)

#dataframe of frequency of docs per topic
topic = []
count_of_docs = []
for i in range(number_of_topics):
    topic.append(i)
    count_of_docs.append(sum(dfdoc["Topic"]==i))

dffreq = pd.DataFrame()
dffreq["Topic"] = topic
dffreq["Docs Count"] = count_of_docs
dffreq.to_csv("lda_docsfreq3.csv",index=False)

#plot the frequency chart
dffreq.plot.bar(x="Topic",y="Docs Count",title="Topics per document frequency(LDA)")
plt.show()

diff_df = dfdoc[dfdoc['diff'].isnull()]
diff_df.to_csv("Zero_diffx.csv",index=False)#len 263

#retrieve the document numbers which have 0 probs diff
x = dfdoc[dfdoc['diff'].isnull()]["Document"].reset_index(drop=True)

##again writing the data into the same file

df_new = pd.read_csv("final_tweets_02_date.csv")#len=2827
df_new["Index"] = list(range(0, len(df_new)))#make an index column

#removing those tweets
for i in x:
    #print(df_new.iloc[i,4])
    df_new.drop(df_new[df_new['Index']==i].index, inplace = True)
    
df_new = df_new.reset_index(drop=True)
df_new["Index"] = list(range(0,len(df_new)))#len = 2564

#applying LDA model again
# the vectorizer object will be used to transform text to vector form
tf_vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')

# apply transformation(convert it to string first)
tf = tf_vectorizer.fit_transform(df_new['clean_tweet'].values.astype('U'))

# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = tf_vectorizer.get_feature_names()

number_of_topics = 10

model1 = LatentDirichletAllocation(n_components=number_of_topics,random_state=0)
model1.fit(tf)

table1 = display_topics(model1, tf_feature_names, no_top_words)
table1.to_csv('results_lda4.csv',index=False)

#gives the document term matrix
doc_topic = model1.transform(tf)
lda_doc_topic = pd.DataFrame(doc_topic)
lda_doc_topic.to_csv('lda_doctop4.csv',index = False)

doc = []
topic = []
tweets = []
for n in range(lda_doc_topic.shape[0]):
    doc.append(n)
    topic_most_pr = doc_topic[n].argmax()
    topic.append(topic_most_pr)
    tweets.append(df_new.iloc[n,3])
    #print("doc: {} topic: {}\n".format(n,topic_most_pr))
    
dfdoc = pd.DataFrame()
dfdoc["Topic"] = topic
dfdoc["Topic prob"] = lda_doc_topic.max(axis=1)
    
maxes = lda_doc_topic.max(axis=1)
less_than_max = lda_doc_topic.where(lda_doc_topic.lt(maxes, axis='rows'))
seconds = less_than_max.max(axis=1)
dfdoc['diff'] = maxes - seconds

dfdoc["Document"] = doc
dfdoc["Tweet"] = tweets
dfdoc = dfdoc.sort_values(by=["Document"])
dfdoc.to_csv("lda_doc4.csv",index=False)

y = dfdoc[dfdoc['diff'].isnull()]["Document"].reset_index(drop=True)#empty :)
####------------------------------------###
#sum(round(dfdoc["Topic prob"],1)<0.5) = 698(should decrease)
#sum(round(dfdoc["diff"],1)>=0.5) = 1198(should increase)

#dataframe of frequency of docs per topic
topic = []
count_of_docs = []
for i in range(number_of_topics):
    topic.append(i)
    count_of_docs.append(sum(dfdoc["Topic"]==i))

dffreq = pd.DataFrame()
dffreq["Topic"] = topic
dffreq["Docs Count"] = count_of_docs
dffreq.to_csv("lda_docsfreq4.csv",index=False)

#plot the frequency chart
dffreq.plot.bar(x="Topic",y="Docs Count",title="Topics per document frequency(LDA)_1")
plt.show()

diff_df = dfdoc[dfdoc['diff'].isnull()]
diff_df.to_csv("Zero_diffy.csv",index=False)#empty
############################################################################3
#applying NMF model
number_of_topics = 10

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=25,token_pattern='\w+|\$[\d\.]+|\S+')

tfidf = tfidf_vectorizer.fit_transform(df['clean_tweet'].values.astype('U'))

tf_feature_names1 = tfidf_vectorizer.get_feature_names()

model2 = NMF(n_components=number_of_topics, random_state=0, alpha=.1, l1_ratio=.5)

model2.fit(tfidf)

table2 = display_topics(model2,tf_feature_names1, no_top_words)

table2.to_csv("results_nmf3.csv",index=False)

#gives the document term matrix
doc_topic = model2.transform(tfidf)
nmf_doc_topic = pd.DataFrame(doc_topic)
nmf_doc_topic.to_csv('nmf_doctop3.csv',index = False)

doc = []
topic = []
tweets = []
for n in range(nmf_doc_topic.shape[0]):
    doc.append(n)
    topic_most_pr = doc_topic[n].argmax()
    topic.append(topic_most_pr)
    tweets.append(df.iloc[n,3])
    #print("doc: {} topic: {}\n".format(n,topic_most_pr))
    
dfdoc = pd.DataFrame()
dfdoc["Topic"] = topic
dfdoc["Topic prob"] = nmf_doc_topic.max(axis=1)

maxes = nmf_doc_topic.max(axis=1)
less_than_max = nmf_doc_topic.where(nmf_doc_topic.lt(maxes, axis='rows'))
seconds = less_than_max.max(axis=1)
dfdoc['diff'] = maxes - seconds

dfdoc["Document"] = doc
dfdoc["Tweet"] = tweets
dfdoc = dfdoc.sort_values(by=["Document"])
dfdoc.to_csv("nmf_doc3.csv",index=False)

#sum(round(dfdoc["Topic prob"],1)<0.5) = 888(should decrease)
#sum(round(dfdoc["diff"],1)>=0.5) = 822(should increase)

#dataframe of frequency of docs per topic
topic = []
count_of_docs = []
for i in range(number_of_topics):
    topic.append(i)
    count_of_docs.append(sum(dfdoc["Topic"]==i))

dffreq = pd.DataFrame()
dffreq["Topic"] = topic
dffreq["Docs Count"] = count_of_docs
dffreq.to_csv("nmf_docsfreq3.csv",index=False)

diff_df = dfdoc[dfdoc['diff'].isnull()]
diff_df.to_csv("Zero_diff1.csv",index=False)
###################
#result = found that a string "via namo app" has the highest frequency for which the result doesnt show
#nything relevant topic. so we need to remove the string to improve the result