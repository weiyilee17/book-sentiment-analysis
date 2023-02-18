from streamlit import title, subheader, plotly_chart
from nltk import download
from nltk.sentiment import SentimentIntensityAnalyzer
from glob import glob
from pathlib import Path
from plotly.express import line

download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

file_paths = glob('diary/*.txt')

analyze_result = []

for single_file_path in file_paths:
    with open(single_file_path, 'r') as file:
        content = file.read()
        score = analyzer.polarity_scores(content)
        score['date'] = Path(single_file_path).stem
        analyze_result.append(score)

analyze_result.sort(key=lambda single_score: single_score['date'])

title('Diary Tone')

subheader('Positivity')

positivity_figure = line(data_frame=analyze_result, x='date', y='pos', labels={
    'date': 'Date',
    'pos': 'Positivity'
})
plotly_chart(positivity_figure)

subheader('Negativity')

negativity_figure = line(data_frame=analyze_result, x='date', y='neg', labels={
    'date': 'Date',
    'neg': 'Negativity'
})
plotly_chart(negativity_figure)
