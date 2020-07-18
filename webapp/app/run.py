import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
#from sklearn.externals import joblib
import joblib # this is modified to allow the file to run on a newer sklearn version
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    graph_one = []
    graph_one.append(
          Bar(
          x = genre_names,
          y = genre_counts,
          name='Genre Counts'
          ))
    layout_one = dict(title = 'Message Counts by Genre',
                yaxis=dict(
                    title='Number of Messages',
                    titlefont_size=16,
                    tickfont_size=20,
                ),
                xaxis=dict(
                    title='Genre',
                    titlefont_size=16,
                    tickfont_size=20,
                ), 
            )


    # add second visual
    category_columns = df.iloc[:,4:]
    categories = list(category_columns.columns.values)[1:]
    category_counts = category_columns.iloc[:,1:].sum().values.tolist() 
    graph_two = []
    graph_two.append(
        Bar(
            x = categories,
            y = category_counts
        )
    )
    layout_two = dict(title = 'Message Counts by Category',
                yaxis=dict(
                    title='Number of Messages',
                    titlefont_size=16,
                    tickfont_size=20,
                ),
                xaxis=dict(
                    titlefont_size=16,
                    tickfont_size=20,
                ), 
            )

    # add third visual
    rowSums = category_columns.iloc[:,1:].sum(axis=1) # not counting related category
    multiLabel_counts = rowSums.value_counts() # count the number of messages with X number of labels
    graph_three = []
    graph_three.append(
        Bar(
            x = multiLabel_counts.index, 
            y = multiLabel_counts.values
        )

    )
    layout_three = dict(title = 'Message Counts by Number of Labels',
                yaxis=dict(
                    title='Number of Messages',
                    titlefont_size=16,
                    tickfont_size=20,
                ),
                xaxis=dict(
                    title='Number of Labels for Message',
                    titlefont_size=16,
                    tickfont_size=20,
                ), 
            )

    # add fourth visual
    correlation_df = category_columns.iloc[:,1:].corr() # exclude child alone column
    graph_four = []
    graph_four.append(
        Heatmap(
            z = correlation_df,
            x=categories,
            y= categories
        )

    )
    layout_four = dict(title = 'Correlation between Categories',
            )


    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html',
                        ids=ids,
                        graphJSON=figuresJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host='localhost', debug=True)

if __name__ == '__main__':
    main()