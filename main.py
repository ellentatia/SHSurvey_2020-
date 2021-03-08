"""
This script was written for visualization of the stakeholders data set survey 2020
by the IPBES Technical support unit on knowledge and data

Developer: Ellen Guimaraes 
Contact: ellenguimaraes@gmail.com
         ellen.guimaraes@senckenberg.de
last update: 08.03.2021
"""
import pandas as pd     #(version 1.0.0)
import plotly           #(version 4.5.4) pip install plotly==4.5.4
import plotly.express as px
import dash             #(version 1.9.1) pip install dash==1.9.1
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
import numpy as np
import dash_table
import plotly.graph_objects as go
import dash_auth

#---------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])
server = app.server

auth = dash_auth.BasicAuth(
    app,
    {'ipbes':'tsudata',
      'joy': 'science',
     'aidin': 'head'}
)

#-----------------------------------------------------------------
app.title = 'SHSurvey_2020'

#Data set
df = pd.read_csv("df_shs_regions.csv", sep ='\t', index_col= False)
regions_all= ['African States',
 'Asia-Pacific States',
 'Eastern European States',
 'Latin American and Caribbean States',
 'Western European and other States']

countries_all = sorted(df['Country of professional activity'].unique())

list_tools=['Capacity-building webpages',
       'Catalogue of assessments',
       'Events / calendar',
       'Impact tracking database',
       'IPBES Assessment full reports',
       'IPBES Assessment Summaries for Policymakers',
       'IPBES Plenary decisions',
       'Meeting documents (e.g. MEP, Bureau, Task Forces)',
       'Website ',
       'IPBES online conferences',
       'Policy support portal',
       'IPBES social media channels',
       'Stakeholder webpages',
       'IPBES webinars',
       'IPBES E-Learning ',
       'Guide on the production of assessments']

list_use=['Policy and/or decision-making',
       'Curriculum, teaching, training',
       'Reports and projects',
       'Research activities',
       'Development of new research priorities',
       'Fund raising and/or resource mobilisation',
       'Disseminate IPBES findings and tools',
       'Organize IPBES uptake events']

#ipbes logo
image_filename = 'Image.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
#---------------------------------------------------------------

card_responses = dbc.Card(
                    dbc.CardBody([
                        html.H4("Total Responses:", className="card-title", style={'color':'Black'}),
                        html.P(id='Total Responses', style={'fontSize':20, 'textAlign': 'left', 'color':'#0072B2'}),

                        html.P(id='Per choice Responses', style={'fontSize':15, 'textAlign': 'left', 'color':'Blue'})
                        ]
                    ),outline = True, color="light"),
card_dropdown = dbc.Card(
                    dbc.CardBody(
                        [
                        html.H5("Regions & Countries: ", className="card-title"),
                        dbc.Row([dbc.Col([html.P("Choose Region:", style={'fontSize':15, 'textAlign': 'center'}),
                        dcc.Dropdown(id='region_dropdown',
                                     options=[{'label': s, 'value': s} for s in regions_all],
                                     placeholder ="All Regions",
                                     clearable= True
                                     )]),
                        dbc.Col([html.P("Choose Country:", style={'fontSize':15, 'textAlign': 'center'}),
                        dcc.Dropdown(id='country_dropdown',
                                     options=[],
                                     clearable = True,
                                     value=[],
                                     placeholder ="Country(s)",
                                     )])])]), color="light"),
app.layout = html.Div([

    dbc.Row([dbc.Col(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '150px'}),
    width={'size': 6, 'offset': 0},
                 ), ]),
    dbc.Row([dbc.Col(html.H3("IPBES Stakeholder survey 2020", style={'fontSize':30, 'textAlign':'center', 'text-decoration': 'bold'}),
                    width={'size': 6, 'offset': 2},
                    ),
            dbc.Col(card_responses, width= 4)
            ]),
    html.Br(),

    dbc.Row(dbc.Col(card_dropdown, width =12), justify ='center'),
    html.Br(),
    dbc.Row([dbc.Col(id='graph6 container', children=[], width = 4), dbc.Col(id='graph container', children=[], width = 7)], justify ='center'),
    html.Br(),
    dbc.Row([dbc.Col(id='graph3 container', children=[], width = 3), dbc.Col(id='graph4 container', children=[], width = 4), dbc.Col(id='graph2 container', children=[], width = 4 )],justify ='center'),
    html.Div(dbc.Row(dbc.Col(id='table1 container', children=[])))])

#---------------------------------------------------------------
@app.callback(
    Output('region_dropdown', 'options'),
    Input('country_dropdown', 'value'),
)
def set_region_options(chosen_country):
    if not chosen_country:
        region_of_country = [{'label': s, 'value': s} for s in regions_all]
    else:
        region_of_country = [{'label': c, 'value': c} for c in df['UN Region'][df['Country of professional activity'] == chosen_country].unique()]
    return region_of_country

@app.callback(
    Output('country_dropdown', 'options'),
    Input('region_dropdown', 'value'),
)
def set_countries_options(chosen_region):
    if not chosen_region:
        countries_of_regions = [{'label': s, 'value': s} for s in countries_all]
    else:
        countries_of_regions = [{'label': c, 'value': c} for c in sorted(df['Country of professional activity'][df['UN Region'] == chosen_region].unique())]
    return countries_of_regions

@app.callback(
    Output('Total Responses', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
)
def set_total_stakeholders(selected_country, selected_region):
    if not selected_region and not selected_country:
        total = len(df)
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        total = len(dfr)
    elif selected_region and len(selected_country)!=0:
        dff = df[df['Country of professional activity'] == selected_country]
        total = len(dff)
    elif not selected_region and len(selected_country)!= 0:
        dff = df[df['Country of professional activity'] == selected_country]
        total = len(dff)
    return '{} Stakeholders'.format(total)

@app.callback(
    Output('Per choice Responses', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
)
def set_res_stakeholders(selected_country, selected_region):
    if not selected_region and not selected_country:
        value = 100
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        value = round(len(dfr)/1024*100,2)
    elif selected_region != 0 and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        value = round(len(dff)/1024*100,2)
    elif not selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        value = round(len(dff)/1024*100,2)
    return '{}  % of of the responses '.format(value)

@app.callback(
    Output('graph container', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
    #prevent_initial_call=True
    )
def update_graph(selected_country, selected_region):
    if not selected_region and not selected_country:
        df_tools = df.iloc[:, 14:30]
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        df_tools = dfr.iloc[:, 14:30]
    elif selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_tools = dff.iloc[:, 14:30]
    elif not selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_tools = dff.iloc[:, 14:30]

    df_tools.columns = list_tools
    df_m = df_tools.melt(
        var_name="IPBES tools, resources and products",
        value_name="Frequency")
    df_m['Count'] = 1
    df2 = df_m.groupby(['IPBES tools, resources and products', 'Frequency'])[
        'Count'].sum().to_frame().reset_index().sort_values(by='Count')
    df2['% of Stakeholder'] = round(df2['Count'] / len(df_tools) * 100, 2)

    fig = px.bar(df2.sort_values(by=['% of Stakeholder'], ascending=False), y='IPBES tools, resources and products', x="% of Stakeholder",
                   color='Frequency',
                   barmode="group", orientation='h',
                   hover_data={"Count": True,
                               "Frequency": False,
                               "% of Stakeholder": True,
                               "IPBES tools, resources and products": False},
                   color_discrete_map={
                        "Never": "#cc79a7",
                        "Once or twice": "#0072b2",
                        'Once or several times a month': "#f0e442",
                        'Several times': "#009e73"},
                   category_orders={
                        "Frequency": ['Never', 'Once or twice', 'Once or several times a month', 'Several times'],
                     }, height=600, width=850)

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="right",
        x=1.05
    ), barmode='group',
    title='Frequency of use of IPBES tools, resources and products:',
    plot_bgcolor = 'rgba(0,0,0,0)')

    return dcc.Graph(id='display-horizbar', figure=fig)

@app.callback(
    Output('graph2 container', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
    #prevent_initial_call=True
    )
def update_graph2(selected_country, selected_region):
    if not selected_region and not selected_country:
        df_C_tools_uses = df.iloc[:,30:38]
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        df_C_tools_uses = dfr.iloc[:,30:38]
    elif selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_C_tools_uses = dff.iloc[:, 30:38]
    elif not selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_C_tools_uses = dff.iloc[:,30:38]

    c = [df_C_tools_uses[x].count() for x in df_C_tools_uses.columns]
    d = [round(df_C_tools_uses[x].count() / sum(c) * 100, 2) for x in df_C_tools_uses.columns]
    df_use_tools = pd.DataFrame(list(zip(list_use, c, d, )),
                                columns=['Activities Used IPBES tools, resources and products', 'Total', 'total answers for activity / total answers %'])
    df_final = df_use_tools[df_use_tools['Total'] > 0]
    fig = px.bar(df_final.sort_values(by='total answers for activity / total answers %')[:6], x ='total answers for activity / total answers %',
                 y='Activities Used IPBES tools, resources and products',
                 text="total answers for activity / total answers %",
                 hover_data={'total answers for activity / total answers %':False,
                             'Activities Used IPBES tools, resources and products': False,
                             'Total': True})

    fig.update_traces(marker_color="#0072B2")

    fig.update_layout(title='Use of IPBES tools, resources and products:',plot_bgcolor = 'rgba(0,0,0,0)')

    return dcc.Graph(id='display-bar', figure=fig)

@app.callback(
    Output('graph3 container', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
    #prevent_initial_call=True
    )
def update_graph3(selected_country, selected_region):
    if not selected_region and not selected_country:
        duration = df.iloc[:, 39].copy()
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        duration = dfr.iloc[:, 39].copy()
    elif selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        duration = dff.iloc[:, 39].copy()
    elif not selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        duration = dff.iloc[:, 39].copy()

    D = duration.value_counts(normalize=True)
    d = (100 * D).round(1)
    E = duration.value_counts()
    df_years = pd.DataFrame(list(zip(d.index, d.values, E.values)),
                                columns=['Years', '% Stakeholders', 'total'])

    fig = px.bar(df_years.sort_values(by= '% Stakeholders'), y='Years',
                 x='% Stakeholders', text="% Stakeholders",
                 hover_data={'% Stakeholders': True,
                             'Years': False,
                             'total': True}
                 )
    fig.update_traces(marker_color="#0072B2")

    fig.update_layout(title='Longevity of engagement with IPBES:', plot_bgcolor = 'rgba(0,0,0,0)')

    return dcc.Graph(id='display-bar3', figure=fig)

@app.callback(
    Output('graph4 container', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
    #prevent_initial_call=True
    )
def update_graph4(selected_country, selected_region):
    if not selected_region and not selected_country:
        capacity = df.iloc[:, 38].copy()
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        capacity = dfr.iloc[:, 38].copy()
    elif selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        capacity = dff.iloc[:, 38].copy()
    elif not selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        capacity = dff.iloc[:, 38].copy()

    A = capacity.value_counts(normalize=True)
    a = round(A * 100, 2)
    B = capacity.value_counts()
    df_capacity = pd.DataFrame(list(zip(a.index, a.values, B.values)),
                            columns=['Capacity', '% Stakeholders', 'Total'])

    fig = px.bar((df_capacity.sort_values(by= '% Stakeholders', ascending=False)[:6]).sort_values(by='% Stakeholders'), y='Capacity',
                 x='% Stakeholders',
                 text='% Stakeholders',
                 hover_data={'Total': True,
                             'Capacity': False,
                             '% Stakeholders': True}
                 )
    fig.update_traces(marker_color="#0072B2")
    fig.update_layout(title='Capacity in which respondents engage with IPBES:', plot_bgcolor = 'rgba(0,0,0,0)')

    return dcc.Graph(id='display-bar4', figure=fig)

@app.callback(
    Output('graph6 container', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
    #prevent_initial_call=True
    )
def update_graph6(selected_country, selected_region):
    if not selected_region and not selected_country:
        df_s_a = df[['Age category','Gender']].copy()
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        df_s_a = dfr[['Age category','Gender']].copy()
    elif selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_s_a = dff[['Age category','Gender']].copy()
    elif not selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_s_a = dff[['Age category','Gender']].copy()

    df_s_a.columns = ['Age category','Sex']
    df_sex_age = df_s_a[df_s_a['Age category'] != 'I do not want to answer'].copy()
    df_sex_age = df_sex_age[df_sex_age['Sex'] != 'I do not want to answer']
    df_sex_age['Count'] = 1
    df_sex_age_ = df_sex_age.groupby(['Age category', 'Sex'])['Count'].sum().to_frame().reset_index()

    fig = px.bar(df_sex_age_, y='Count', x="Age category",
                 barmode="stack",
                 text='Count',
                 color_discrete_map={
                     'Male': 'rgba(70, 192, 193)',
                     'Female': 'rgba(70, 109, 193)',
                   },
                 color='Sex',
                 category_orders={
                     "Sex": ['Male', 'Female']})

    fig.update_layout(legend_traceorder="reversed")

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=sorted(df_sex_age['Age category'].unique()),
            ticktext=['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75+']
        ),
        title='Age and sex of respondents:', plot_bgcolor = 'rgba(0,0,0,0)'
    )

    return dcc.Graph(id='display-stack', figure=fig)

@app.callback(
    Output('table1 container', 'children'),
    Input('country_dropdown', 'value'),
    Input('region_dropdown', 'value'),
    #prevent_initial_call=True
    )

def update_table(selected_country, selected_region):
    if not selected_region and not selected_country:
        df_table = df.iloc[:,116:126].copy()
    elif selected_region and not selected_country:
        dfr = df[(df['UN Region'] == selected_region)]
        df_table = dfr.iloc[:,116:126].copy()
    elif selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_table = dff.iloc[:,116:126].copy()
    elif not selected_region and len(selected_country) != 0:
        dff = df[df['Country of professional activity'] == selected_country]
        df_table = dff.iloc[:,116:126].copy()

    df_table.columns = ['Tele-/video-conference',
                        'Email',
                        'In-person/face-to-face communication',
                        'Facebook',
                        'Twitter',
                        'Linkedin',
                        'YouTube',
                        'Instagram',
                        'Webinar',
                        'IPBES website']
    column_order=['Methods of communication', 'Strongly prefer', 'Somewhat prefer', 'Neither prefer nor dislike',
                      'Somewhat dislike', 'Strongly dislike']
    df_m = df_table.melt(
        var_name="Methods of communication",
        value_name="Preference")
    df_m['Count'] = 1
    df2 = df_m.groupby(['Methods of communication', 'Preference'])['Count'].sum().to_frame().reset_index()
    df2['% of Stakeholder'] = round(df2['Count'] / len(df_table) * 100, 2)
    df2['% of Stakeholder'] = df2['% of Stakeholder'].astype(str) + ' %'
    df_p = df2.pivot(index='Methods of communication', columns='Preference', values='% of Stakeholder').reset_index()
    df_table_=pd.DataFrame()
    for x in column_order:
            if x in df_p.columns:
                df_table_[x] = df_p[x].copy()

    fig = go.Figure(data=[go.Table(
        columnwidth=100,
        header=dict(
            values=list(df_table_.columns),
            fill_color='#0072B2',
            font_size=15,
            font_color='white',
            height=40,
            align='center'),
        cells=dict(
            values=[df_table_[column] for column in df_table_.columns],
            fill_color='lavender',
            align='center',
            font_size=12,
            height=30,
        ))
    ])

    fig.update_layout(title ='Methods of communication favored by respondents:')

    return dcc.Graph(id='table1', figure=fig)
#---------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=False) # changed to false due to the authentication
    #app.run_server(debug=True)
