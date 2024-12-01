import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

# Загрузка данных
sub_data = pd.read_csv('../data/submissions/ml_pandas_submission.csv')
sub_data["ID"] = sub_data.index

# Инициализация приложения
app = dash.Dash(__name__)
app.title = "Автоматизация выдачи ипотеки"

# Основной макет приложения
app.layout = html.Div([
    html.Div(className="tab-header", children=[
        html.H1("Автоматизация выдачи ипотеки"),
        html.Div("ML-панды представляют"),
    ]),
    dcc.Tabs(id='tabs', value='overview', className='tabs', children=[
        dcc.Tab(label='Обзор', value='overview', className='tab'),
        dcc.Tab(label='Детали сделки', value='deal_details', className='tab'),
        dcc.Tab(label='Симулятор', value='simulator', className='tab'),
    ]),
    html.Div(id='tabs-content', className="tab-content")
])

# Логика обработки вкладок
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'overview':
        # Распределение цены недвижимости (фактическая стоимость)
        price_dist = px.histogram(
            sub_data, x='__price_predict', title="Распределение фактической стоимости недвижимости",
            color_discrete_sequence=['#de5571']
        )
        price_dist.update_layout(plot_bgcolor='#1E1E1E', paper_bgcolor='#1e1e2f', font_color='#ffffff')

        # Распределение задолженности (churn)
        churn_dist = px.histogram(
            sub_data, x='__churn_prob', title="Распределение задолженности по ипотеке (churn)",
            color_discrete_sequence=['#de5571']
        )
        churn_dist.update_layout(plot_bgcolor='#1E1E1E', paper_bgcolor='#1e1e2f', font_color='#ffffff')

        return html.Div([
            html.H2("Общая статистика"),
            dcc.Graph(figure=price_dist),
            dcc.Graph(figure=churn_dist),
        ])

    elif tab == 'deal_details':
        return html.Div([
            html.H2("Поиск по ID сделки"),
            dcc.Input(id='deal-id', type='number', placeholder="Введите ID сделки"),
            html.Button("Показать", id='show-deal-btn'),
            html.Div(id='deal-info'),
        ])

    elif tab == 'simulator':
        return html.Div([
            html.H2("Симулятор выдачи ипотеки"),
            dcc.Input(id='input-price', type='number', placeholder="Стоимость недвижимости (млн)"),
            dcc.Input(id='input-income', type='number', placeholder="Ежемесячный доход (тыс)"),
            dcc.Input(id='input-priority', type='number', placeholder="Приоритет сделки"),
            html.Button("Проверить", id='simulate-btn'),
            html.Div(id='simulation-result'),
        ])


# Обработка деталей сделки
@app.callback(
    Output('deal-info', 'children'),
    [Input('show-deal-btn', 'n_clicks'),
     Input('deal-id', 'value')]
)

def show_deal_info(n_clicks, deal_id):
    if not n_clicks or not deal_id:
        return html.Div("", className="result-block")
    deal = sub_data[sub_data['ID'] == deal_id]
    priorities = np.argsort(-sub_data["__priority"])
    if deal.empty:
        return html.Div(f"Сделка с ID {deal_id} не найдена.", className="result-block")
    return html.Div([
        html.H3(f"Информация о сделке {deal_id}"),
        html.P(f"Фактическая стоимость недвижимости: {deal['__price_predict'].values[0]:.4f} млн"),
        html.P(f"Вероятность невозврата (churn): {deal['__churn_prob'].values[0]:.4f}"),
        html.P(f"Приоритет слелки: {int(np.where(priorities == deal_id)[0])}"),
    ], className="result-block")

# Обработка симуляции
@app.callback(
    Output('simulation-result', 'children'),
    [Input('simulate-btn', 'n_clicks'),
     Input('input-price', 'value'),
     Input('input-income', 'value'),
     Input('input-priority', 'value')]
)
def simulate_mortgage(n_clicks, price, income, priority):
    if not n_clicks:
        return html.Div("", className="result-block")
    if price and income and priority:
        approval = "Да" if income > (price * 10) and priority > 0 else "Нет"
        if (approval == "Да"):
            return html.Div(f"Ипотека одобрена", className="result-block")
        return html.Div(f"Заявка отклонена", className="result-block")
    return html.Div("Некорректные данные.", className="result-block")

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)
