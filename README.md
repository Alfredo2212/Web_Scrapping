This is a lead enrichment tool that:
1. Scrapes company websites from google search results using serpAPI
2. Extracts trending keywords using NLP keyword extraction
3. Summarizes interest signals for each company
4. Provides visual insights and keyword frequency charts
5. Exportable excel report

Setup :
1. Clone Repository
>> on your TERMINAL, it is best to create new environment to avoid version crash
2. conda create -n streamlit_env python=3.10
3. conda activate streamlit_env
4. Install required dependencies
   Dependencies list :
   pip install streamlit beautifulsoup4 requests nltk matplotlib pandas
   pip install tldextract rake-nltk
5. run the app with streamlit run app.py

Streamlit Local :
1. use the top left ">" fill the textbox as instructed
2. press run analysis button
3. optional to download the report at the end of analysis

Important Note :
you can find your serpAPI key here
https://serpapi.com/
