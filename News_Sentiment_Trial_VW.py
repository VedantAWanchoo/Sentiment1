

# News Sentiment File

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from newsapi import NewsApiClient
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

# ---------- Utility Functions ----------

def load_model():
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    classifier = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
    return classifier

def sentiment_color(sentiment):
    if sentiment == 'Positive': return 'green'
    elif sentiment == 'Negative': return 'red'
    elif sentiment == 'Neutral': return 'grey'
    return 'white'

def fetch_articles(company):
    NAP_API_KEY = 'f423323bd24f4585ac74596d0e52d282'
    newsapi = NewsApiClient(api_key=NAP_API_KEY)

    current_date = datetime.now()
    past_date = current_date - timedelta(days=30)

    all_info = newsapi.get_everything(
        q=company,
        sources='bbc-news,financial-post,bloomberg,business-insider,reuters,the-wall-street-journal',
        language='en',
        from_param=past_date.strftime('%Y-%m-%d'),
        to=current_date.strftime('%Y-%m-%d'),
        sort_by='publishedAt'
    )

    return all_info["articles"]

# ---------- Streamlit UI ----------

def display_news_sentiment():
    st.title('Company News Sentiment Analysis')
    company = st.text_input('Enter a company name:')

    if company:
        articles = fetch_articles(company)
        classifier = load_model()

        if articles:
            sentiment_counts = {'date': [], 'positive': [], 'negative': [], 'neutral': []}
            results = []

            for article in articles:
                headline = article['title']
                description = article['description']
                combined_text = f"{headline}. {description}"
                result = classifier(combined_text)[0]
                sentiment = result['label']
                date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').date()

                if date not in sentiment_counts['date']:
                    sentiment_counts['date'].append(date)
                    sentiment_counts['positive'].append(0)
                    sentiment_counts['negative'].append(0)
                    sentiment_counts['neutral'].append(0)

                idx = sentiment_counts['date'].index(date)
                if sentiment == 'Positive':
                    sentiment_counts['positive'][idx] += 1
                elif sentiment == 'Negative':
                    sentiment_counts['negative'][idx] += 1
                else:
                    sentiment_counts['neutral'][idx] += 1

                results.append((article, sentiment))

            df = pd.DataFrame(sentiment_counts)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%B %d %Y')
            df_long = df.melt(id_vars='date', var_name='Sentiment', value_name='Count')

            color_mapping = {'positive': 'Green', 'neutral': 'Grey', 'negative': 'Red'}

            st.subheader('Sentiment over time')
            fig = px.bar(df_long, x='date', y='Count', color='Sentiment', barmode='stack',
                         color_discrete_map=color_mapping,
                         labels={'date': 'Date', 'Count': 'Number of Articles', 'Sentiment': 'Sentiment'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f'{len(results)} News Articles Analysed')
            for article, sentiment in results:
                headline = article['title']
                description = article['description']
                source_name = article['source']['name']
                url = article['url']
                timestamp = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').strftime('%B %d, %Y %H:%M')
                color = sentiment_color(sentiment)

                st.markdown(f"""
                <div style="border:1px solid {color}; padding:10px; margin:10px 0;">
                    <strong style="color:{color};">{headline}</strong><br>
                    <span style="color:gray; font-size:small;">{source_name} - {timestamp}</span><br>
                    {description}<br>
                    <strong style="color:{color};">Sentiment: {sentiment}</strong>
                </div>
                """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    display_news_sentiment()


