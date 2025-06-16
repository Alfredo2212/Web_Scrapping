import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import re
import requests
import tldextract
import streamlit as st
import io

import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

from bs4 import BeautifulSoup
from rake_nltk import Rake
from collections import Counter

### STREAMLIT INIT
# Configure user inputs
st.sidebar.title("Search Configuration")

serpapi_key = st.sidebar.text_input("Please input your SerpAPI Key")
if not serpapi_key:
  st.sidebar.warning("Get serpAPI keys here https://serpapi.com/")

company_query = st.sidebar.text_input("What companies or topics are you searching for?")

interest_query = st.sidebar.text_input("Enter a keyword you're curious about (e.g., sustainability):")

num_query = st.sidebar.number_input("Number of Google Results to Fetch", min_value=10, step= 10, value=20)

st.title("Keywords & Trend Scraper")

st.markdown("""
This tool will :
1. Search companies using your SerpAPI key
2. Extract domains and keywords
3. Analyze how those websites discuss subject of interest
""")

# Debug for dev mode & validation
st.write("**Entered SerpAPI Key:**", serpapi_key[:4] + "..." if serpapi_key else "")
st.write("**Search Query:**", company_query)
st.write("**Subject of Interest:**", interest_query)
st.write("**Number of Queries:**", num_query)

if serpapi_key and company_query and interest_query:
  st.success("Ready to search and analyze!")
else:
  st.warning("Please fill in all fields to begin")

# Initialize dataframe for web scraping result
# Scrape from google and get the results
def get_company_data_from_serpapi(company_query, serpapi_key, num_query=20):
  df = pd.DataFrame(columns=[
      'Company',
      'Domain'
  ]) # reinitialize data frame

  for start in range(0, num_query, 10): #scrape amount
    params = {
        "engine": "google",
        "q": company_query,
        "start": start,
        "api_key": serpapi_key
    }

    response = requests.get("https://serpapi.com/search", params=params)
    results = response.json().get("organic_results", [])

  # Fill DataFrame
    for res in results:
      title = res.get("title")
      link = res.get("link")
      if title and link:
        fill_df = pd.DataFrame([{
            'Company': title,
            'Domain': link
        }])
        df = pd.concat([df, fill_df], ignore_index= True)
  return df


# Filter third party / aggregator websites
third_party_indicator = [
    "top", "best", "leading", "directory", "review", "compare", "list",
    "ranking", "companies", "agencies", "firms", "vendors", "providers",
    "expert", "consultant", "outsource", "services", "evaluations", "insights",
    "buyers-guide","blog","wikipedia","how","developers","linkedin","work","year",
    "country","what","where","who","why","when","guide","news","research","report","insight",
    "magazine","travel","looking","fandom","category","directory","news","website","journal"
    "group","agency","paper","article","instagram","facebook","site","tiktok","video",
    "youtube","reddit","quora"
]

def filter_valid_company_sites(df):
  def company_website(domain, company):
    domain = str(domain).lower()
    company = str(company).lower()

    for tp in third_party_indicator:
      if tp in domain or tp in company:
        return False
    return True

  df = df[df.apply(lambda row: company_website(row['Company'], row['Domain']), axis =1)].reset_index(drop=True)
  df = df.drop_duplicates(subset=['Company'], keep='first').reset_index(drop=True)
  df = df.drop_duplicates(subset=['Domain'], keep='first').reset_index(drop=True)

  return df

# Replace site Titles with exact company name for better presentation
def get_company_name(url):
  extracted = tldextract.extract(url)
  domain_part = extracted.domain
  return domain_part.capitalize()


# Get trending words on websites
def query_keywords(url, company_query, top_n=5):
  try:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Get seen text
    for script in soup(["script", "style"]):
      script.extract()

    # limit seen text to main only
    main = soup.find('main') or soup.find('div', {'id': 'main'}) or soup.find('div', {'id': 'content'})
    text = main.get_text(separator=' ') if main else soup.get_text(separator=' ')

    # extract words using Rake nltk
    rake = Rake()
    rake.extract_keywords_from_text(text)
    ranked_phrases = rake.get_ranked_phrases()

    # Exclude words that are in query
    query_words = set(company_query.lower().split())

    filtered_indices = [
        phrases for phrases in ranked_phrases
        if not any(word in query_words for word in phrases.lower().split())
        and len(phrases.strip().split()) >= 2
    ]

    selected_phrases = filtered_indices[:top_n] if filtered_indices else text[:top_n]

    # Clean the phrase and put in a list for ease of process
    all_words = set()
    for phrase in selected_phrases:
      words = phrase.lower().split()
      if all(word.isalpha() for word in words):
        all_words.add(' '.join(words))

    return list(all_words)

  # error for timeout limit
  except requests.exceptions.ReadTimeout:
    print(f"{url} Unable to Load")
    return []

  except Exception as e:
    print(f"Error fetching {url}: {e}")
    return None

# Clean the output
def remove_repeat(phrase, threshold= 0.5):
  words = phrase.lower().split()
  deduped = [words[0]] if words else []
  for i in range(1, len(words)):
    if words[i] != words[i-1]:
      deduped.append(words[i])
  return ' '.join(deduped)

def analyze_keywords(keyword_df):
  indexed_keywords = []
  cleaned_keywords = []

  for idx, keyword_list in keyword_df['Keywords'].items():
    if isinstance(keyword_list, list):
      for phrase in keyword_list:
        indexed_keywords.append((idx, phrase))
        cleaned = remove_repeat(phrase)
        words = cleaned.lower().split()
        if (
            1 <= len(words) <= 8
            and sum(w.isalpha() for w in words) >= len(words) - 1
        ):
          cleaned_keywords.append(cleaned)

  unique_keywords = list(dict.fromkeys(cleaned_keywords))
  keyword_df = pd.DataFrame(unique_keywords, columns=['Keyword'])

  return keyword_df

# User-driven query of interest
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_user_interest(url, interest_query, window=10):
  try:
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    for script in soup(["script", "style"]):
      script.extract()

    main = soup.find('main') or soup.find('div', {'id':'main'}) or soup.find('div', {'id': 'content'})
    text = soup.get_text(separator =' ')
    text = re.sub(r'\s+', ' ', text).lower()

    words = text.split()
    interest = interest_query.lower()

    for i, word in enumerate(words):
      if re.search(rf'\b{re.escape(interest)}\b', word):
        context = words[i+1:i+1+window]

        cleaned = [
            lemmatizer.lemmatize(w)
            for w in context
            if w not in stop_words and w.isalpha() and len (w) > 2
        ]

        return cleaned
  except Exception as e:
    print(f"Error fetching {url}: {e}")
    return []

# Rank based on interest
def interest_rank(interest_rank_df, interest_col='Interest'):
  all_words = [
      word
      for word_list in df['Interest']
      if isinstance(word_list, list)
      for word in word_list
  ]
  word_counts = Counter(all_words)

  # Summary df
  interest_rank_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])
  interest_rank_df.sort_values(by='Count', ascending= False, inplace= True)
  interest_rank_df.reset_index(drop=True, inplace=True)

  return interest_rank_df

### STREAMLIT BUTTON
# Raw query output to streamlit
if st.button("Run Analysis"):
  df = get_company_data_from_serpapi(company_query, serpapi_key)
  df = filter_valid_company_sites(df)

  # apply found name and clean the domain name
  df['Company'] = df['Domain'].apply(get_company_name)
  df['Domain'] = df['Domain'].str.replace(r'\?.*', '', regex=True)

  # Trending words
  df['Keywords'] = df['Domain'].apply(lambda url: query_keywords(url, company_query))
  keyword_df = analyze_keywords(df)

  # Subject of Interest
  df['Interest'] = df['Domain'].apply(lambda url: extract_user_interest(url, interest_query))
  interest_rank_df = interest_rank(df)

  # Streamlit
  # Raw query
  if not df.empty:
    st.subheader("Raw Query Table")
    st.dataframe(df)
  else:
    st.warning("Please run a search to populate the table")

  # Keyword found output to streamlit
  if not keyword_df.empty:
    st.subheader("Trending keywords")
    st.dataframe(keyword_df)
  else:
    st.warning("No trending keywords found")

  # Interest ranking output to streamlit
  if not interest_rank_df.empty:
    st.subheader(f"Interest dependencies for {interest_query}")

    # Plot the subject of interest for better readibility
    fig, ax = plt.subplots()
    ax.barh(interest_rank_df['Word'], interest_rank_df['Count'], color='skyblue')
    ax.set_xlabel('Count')
    ax.set_ylabel('Words')
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer= True))
    st.pyplot(fig)
  else:
    st.warning("No dependencies found")

  # Summary panel
  st.subheader("Summary Panel")
  with st.container():
    col1, col2= st.columns(2)

    with col1:
      st.metric("Extracted Company Websites", len(df))
      st.metric("Extracted Keywords", df["Keywords"].apply(bool).sum())

    with col2:
      total_interest_tags = interest_rank_df['Count'].sum() if not interest_rank_df.empty else 0
      top_interest = interest_rank_df.iloc[0]['Word'] if not interest_rank_df.empty else "N/A"
      st.metric("Total Interest Tags", total_interest_tags)
      st.metric("Top Interest Tag", top_interest)

  # Export panel
  st.subheader("Export Data")
  if not df.empty:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
      df.to_excel(writer, sheet_name="Raw Query Table", index=False)

      if not keyword_df.empty:
        keyword_df.to_excel(writer, sheet_name='Trending Keywords', index=False)
      else:
        pd.DataFrame(["No Keywords found"]).to_excel(writer, sheet_name='Trending Keywords', index=False, header=False)

      if not interest_rank_df.empty:
        interest_rank_df.to_excel(writer, sheet_name='Interest Ranking', index=False)
      else:
        pd.DataFrame(["No interest dependencies found"]).to_excel(writer, sheet_name='Interest Ranking', index=False, header=False)

    output.seek(0)
    report_name = f"Scraping {company_query} with subject of interest {interest_query}.xlsx"

    st.download_button(
        label="Download Report",
        data=output,
        file_name=report_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
  else:
    st.warning("There is nothing to download, please rerun")



