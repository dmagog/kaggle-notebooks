# Stellar Class EDA: which class breaks your score

Разбор [Playground S6E6 · Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6): по горстке чисел о яркости определить, что за объект — звезда, галактика или квазар. Обучающий разбор от сырого света до честной базовой модели без утечек.

- **Оригинал на Kaggle:** [an-almost-exhaustive-stellar-class-eda](https://www.kaggle.com/code/georgymamarin/an-almost-exhaustive-stellar-class-eda-final) · Notebook 🥉
- **Ноутбук в репозитории:** [stellar-class-eda.ipynb](stellar-class-eda.ipynb)

Что показываю:
- **разделение классов по цветам** (u−g vs g−r) — три популяции расходятся ещё до модели;
- **ловушка высокой взаимной информации** — признак с высоким MI, но нулевым приростом;
- **стратифицированная кросс-валидация без утечек**, матрица ошибок на границе GALAXY↔QSO, базовая модель LightGBM.

<p align="center">
  <img src="color_color.png" width="480" alt="разделение классов по цветам"><br>
  <img src="redshift_by_class.png" width="480" alt="красное смещение по классам">
</p>
