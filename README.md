# Запуск

Python 3.13. Команды из папки `wine_app`:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Кэширование

`st.cache_resource` хранит один объект модели. Без него `joblib.load` читал бы файл заново при каждом движении ползунка. `st.cache_data` возвращает копии таблицы и массива вероятностей из кэша. При смене порога вероятности не пересчитываются.
