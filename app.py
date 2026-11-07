import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title='Оценка вина', layout='wide')
st.title('Оценка вина')

model = load_model()
test = load_test()
X_test = test.drop(columns='target')
y_test = test['target']
columns = X_test.columns.tolist()
default_threshold = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))['threshold']
threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, default_threshold,
                             step=0.001, format='%.4f')

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Загрузка CSV', 'Коэффициенты'
])

with single_tab:
    inputs = {}
    fields = st.columns(3)
    for i, name in enumerate(columns):
        with fields[i % 3]:
            if name == 'is_red':
                inputs[name] = int(st.toggle('Красное вино', value=bool(X_test[name].median()),
                                             help='Если выключено, вино белое.'))
            else:
                inputs[name] = st.number_input(
                    name, min_value=float(X_test[name].min()), max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()), step=0.0001 if name == 'density' else 0.01,
                    format='%.5f' if name == 'density' else '%.3f'
                )
    wine = pd.DataFrame([inputs], columns=columns)
    p = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность оценки 7 и выше', f'{p * 100:.1f}%')
    st.write('Решение:', 'включить в подборку' if p >= threshold else 'не включать в подборку')

with metrics_tab:
    p_test = test_proba(model, X_test)
    pred = p_test >= threshold
    matrix = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    a, b, c = st.columns(3)
    a.metric('Precision', f'{precision:.3f}' if pred.any() else 'нет отбора')
    b.metric('Recall', f'{recall:.3f}')
    c.metric('F1', f'{f1:.3f}')
    st.dataframe(pd.DataFrame(matrix, index=['Факт: 0', 'Факт: 1'],
                              columns=['Прогноз: 0', 'Прогноз: 1']))

    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr, rc, _ = metrics.precision_recall_curve(y_test, p_test)
    fig = Figure(figsize=(10, 4))
    ax_roc, ax_pr = fig.subplots(1, 2)
    ax_roc.plot(fpr, tpr)
    ax_roc.plot([0, 1], [0, 1], '--', color='gray')
    ax_roc.scatter(fp / (fp + tn), tp / (tp + fn), color='tomato', label='Текущий порог')
    ax_roc.set(title='ROC', xlabel='FPR', ylabel='TPR', xlim=(0, 1), ylim=(0, 1.05))
    ax_pr.plot(rc, pr)
    ax_pr.axhline(y_test.mean(), linestyle='--', color='gray')
    ax_pr.scatter(recall, precision if pred.any() else 1.0, color='tomato', label='Текущий порог')
    ax_pr.set(title='PR', xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.05))
    ax_roc.legend()
    ax_pr.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.write(f'В подборку попало {int(pred.sum())} из {len(pred)} вин.')
    st.write('При высоком пороге отбор строже. Если снизить порог, в подборку попадёт '
             'больше вин, в том числе больше вин с оценкой ниже 7.')

with batch_tab:
    st.write('Нужны 12 признаков из `test.csv`. Столбец `target` не нужен.')
    uploaded = st.file_uploader('CSV с признаками', type='csv')
    if uploaded is not None:
        try:
            rows = pd.read_csv(uploaded)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
            st.error('Не удалось прочитать CSV. Проверьте кодировку и разделитель.')
        else:
            missing = [name for name in columns if name not in rows.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            elif rows.empty:
                st.error('В файле нет строк с данными.')
            else:
                features = rows[columns].apply(pd.to_numeric, errors='coerce')
                if not np.isfinite(features.to_numpy(dtype=float)).all():
                    st.error('Признаки должны быть числами без пропусков.')
                elif not features['is_red'].isin([0, 1]).all():
                    st.error('В столбце is_red допустимы только 0 и 1.')
                else:
                    probabilities = model.predict_proba(features)[:, 1]
                    result = rows.assign(probability=probabilities,
                                         prediction=(probabilities >= threshold).astype(int))
                    st.dataframe(result)
                    st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8-sig'),
                                       file_name='wine_predictions.csv', mime='text/csv')

with weights_tab:
    weights = pd.Series(model.steps[-1][1].coef_[0], index=columns).sort_values()
    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()
    ax.barh(weights.index, weights, color=['tomato' if w < 0 else 'steelblue' for w in weights])
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после стандартизации')
    fig.tight_layout()
    st.pyplot(fig)
    st.write('Больше всего повышают оценку модели:', ', '.join(weights.tail(2).index[::-1]))
    st.write('Больше всего снижают оценку модели:', ', '.join(weights.head(2).index))
