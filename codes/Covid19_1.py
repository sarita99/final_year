##this code will help you extract tweets per day(i have done this way) 
##the hashtags or keywords related to coronavirus have been taken from twitter site(a full list of all the keywords were available in the site)
import pandas as pd
import tweepy

#the query length is limited(nearly 500 characters)
covid19_keywords = '#CoronaAlert OR #coronavirusdelhi OR #coronavirusinindia OR #covid_19ind OR #covid19india OR #coronavirusindia OR #QuarantineLife OR #CONVID19 OR #lockdown OR #pandemic OR #quarantine OR #QuarantineAndChill OR #remoteworking OR #socialdistancing OR #stayhome OR #stayhomesavelives OR #workfromhome OR #StaySafeStayHome OR #WorkingFromHome OR #IndiaFightsCorona -filter:retweets'

#Twitter credentials for the app
consumer_key = XXXXXXXXXXX
consumer_secret = XXXXXXXXXXX
access_key= XXXXXXXXXXX
access_secret = XXXXXXXXXXX

#geocode='28.60476,77.38276,50km' for delhi region

#pass twitter credentials to tweepy
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

covid19_tweets= "covid19_tweets_22.csv"

#columns of the csv file
COLS = ['created_at','original_text','user_location']

##pulling tweets of one day
#set two date variables for date range
start_date = '2020-05-22'
end_date = '2020-05-23'

#method write_tweets()
def write_tweets(keyword,file):
    
    df = pd.DataFrame(columns=COLS)
    #page attribute in tweepy.cursor and iteration(until=before the given date)
    for page in tweepy.Cursor(api.search,q=keyword ,geocode='28.60476,77.38276,50km',lang='en',since=start_date,until=end_date,result_type='mixed',count=100,tweet_mode='extended').pages(2000):
        for status in page:
            new_entry = []
            status = status._json
            #new entry append
            new_entry += [status['created_at'],status['full_text'],status['user']['location']]          
            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
                
    df.to_csv(file,columns = COLS, index=False, encoding="utf-8")

#call main method passing keywords and file path
write_tweets(covid19_keywords,covid19_tweets)
