# Bullseye / Bullshit

Fake news classifier including exploration analysis and NLP techniques 🎯

# Project Overview
As part of the Udacity Nanodegree capstone project, the following analysis attempts to distinguish between news headlines that are real and those that are fake.
This post details insights following data exploration as well as the NLP pipeline setup using custom transformers along with hyperparameter tuning using grid search.

Medium post of this project can be found [here](https://medium.com/@plan__b/auto-fake-news-classifier-using-headlines-86c98fa5ab6e).

# Problem statement
Even though disinformation often disguises itself as legitimate information, the assumption of this project is that there are defining stylistic features (such as the use of capitalisation and pronouns) that can be used to distinguish the two types of content.

Headlines are the first content web users are exposed to, most prominently via social media. The aim is to use machine learning to distinguish between fake and real content while identifying prominent characteristics associated with disinformation. The scope of this project therefore does not include other metadata such as article content, date of publication and source.

The problem to be solved is therefore to automatically identify fake news by solely analysing the content headline or title text style. In the context of the project, "fake" news is defined as news that is intentionally meant to mislead based on false, misinterpreted or manipulated facts.

## Data
Two datasets are used in this project. Data exploration and initial model setup testing used a smaller dataset, while a much larger model is ultimately used.

### Covid-19 misinformation
The initial data used for this project was collected by Susan Li in her excellent [project](https://towardsdatascience.com/automatically-detect-covid-19-misinformation-f7ceca1dc1c7) earlier this year used to detect Covid-19 misinformation.

The table consists of the following columns: title, text, source and label. With a total of 1,159 records, there's an almost-even split of 579 "true" articles and 490 "fake" ones.

### Fake News Corpus
Upon expansion of this project, data from [Fake News Corpus](https://github.com/several27/FakeNewsCorpus) was collected. This is an open source dataset composed of millions of news articles mostly scraped from a curated list.

For purposes of this project, 130,443 titles labeled "reliable" and "fake" were extracted and used to further train and test the model.

# Results
The final model has a weighted average F₁ score of 86% using the Covid-19 misinformation dataset.

![Image of Covid-19 Confusion Matrix](https://github.com/feeblefruits/fake_news_classifier/blob/main/assets/covid_confusion.png)

The same model parameters are used to train on Fake News Corpus article titles which exceed 100,000 and scores 97%.

![Image of Fake News Corpus Confusion Matrix](https://github.com/feeblefruits/fake_news_classifier/blob/main/assets/fakenews_corpus_confusion.png)

Upon normalising the dataset, however, a weighted average of 85% is achieved.

## Files

Files included in this project are as follows:
```
data_eploration.ipynb is used for analaysis of the Covid-19 Disinformation dataset
model_pipeline.ipynb is used to setup model pipeline and grid search of above dataset
model_pipeline_updated_data.ipynb includes the same model parameters initialised in the above but trains and tests on a much larger Fake News Corpus dataset
custom_transformers.py includes custom transformers imported to the updated pipeline

data/ includes the Fake News Corpus dataset
assets/ includes images used in this readme

```

## Libraries used

```
Pandas
Numpy
Re
Itertools
Collections
String
BeautifulSoup
Nltk
Sklearn
Spacy
Seaborn
Plotly
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
