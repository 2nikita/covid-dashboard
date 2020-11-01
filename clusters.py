from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

def target_value(x, continents):
    if x == continents[0]:
        return 0
    elif x == continents[1]:
        return 1
    else:
        return 2

def annotate(fig, vars):
    fig.add_annotation(x=0.5, y=0.85, xanchor='left', yanchor='bottom',
                        xref='paper', yref='paper', showarrow=False, align='left',
                        bgcolor='rgba(255, 255, 255, 0.5)',
                        text=str(vars[0]) + ', ' + str(vars[1]) + ', ' + str(vars[2]))

def model_clusters(data, continents, date, vars):
    df1 = data[(data['date'] == date) & (data['continent'].isin(continents))]
    df = df1[['continent', 'location'] + vars].reset_index()
    df['target'] = df['continent'].map(lambda x: target_value(x, continents))
    for var in vars:
        df[var] = np.log(df[var])
    df = df.dropna()

    X = df[vars].to_numpy()
    y = df['target'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1547)

    model = SVC()
    parameters = {'C': np.linspace(0.1, 5.1, 51),  'kernel': ['linear']}
    grid = GridSearchCV(model, parameters, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    y_train_pred = grid.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    y_test_pred = grid.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    y_pred = grid.predict(X)
    accuracy_all = accuracy_score(y, y_pred)

    df['target_pred'] = y_pred

    return {'df': df, 'accuracy_all': round(accuracy_all, 3), 'accuracy_train': round(accuracy_train, 3),
            'accuracy_test': round(accuracy_test, 3)}

def plot_clusters(data, continents, vars, date):
    result = model_clusters(data, continents, date, vars)
    df = result['df']
    fig1 = go.Figure()
    fig2 = go.Figure()

    for i in range(0, len(continents)):
        fig1.add_trace(go.Scatter3d(x=df[df['target'] == i][vars[0]],
                                    y=df[df['target'] == i][vars[1]],
                                    z=df[df['target'] == i][vars[2]],
                                    mode='markers', marker_size=5,
                                    name=continents[i]))

        fig2.add_trace(go.Scatter3d(x=df[df['target_pred'] == i][vars[0]],
                                    y=df[df['target_pred'] == i][vars[1]],
                                    z=df[df['target_pred'] == i][vars[2]],
                                    mode='markers', marker_size=5,
                                    name=continents[i]))

    camera = dict(
        eye=dict(x=0.5, y=2, z=0.8)
    )

    fig1.update_layout(title='Clustering by regions: actual data', title_x=0.5, autosize=True, showlegend=False,
                      scene=dict(xaxis=dict(title=vars[0], titlefont_color='white'),
                                 yaxis=dict(title=vars[1], titlefont_color='white'),
                                 zaxis=dict(title=vars[2], titlefont_color='white')),
                      scene_camera=camera, template='plotly_dark', margin={'l': 20, 'b': 15, 'r': 10, 't': 50}, hovermode='closest')
    annotate(fig1, vars)

    fig2.update_layout(title='Clustering by regions: predicted data', title_x=0.17, autosize=True, showlegend=True,
                      scene=dict(xaxis=dict(title=vars[0], titlefont_color='white'),
                                 yaxis=dict(title=vars[1], titlefont_color='white'),
                                 zaxis=dict(title=vars[2], titlefont_color='white')),
                      scene_camera=camera, template='plotly_dark', margin={'l': 20, 'b': 15, 'r': 10, 't': 50})

    annotate(fig2, vars)

    return fig1, fig2, result['accuracy_all'], result['accuracy_train'], result['accuracy_test']