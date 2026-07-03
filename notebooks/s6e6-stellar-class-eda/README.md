# An almost-exhaustive Stellar Class EDA

**Соревнование:** [Playground Series S6E6 — Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6) · закрыто.
**Итог:** Notebook 🥉 (бронза за апвоты). Playground competition-медалей не даёт.
**Жанр:** обучающий accessibility-first EDA.

Разбор классификации спектрального класса объекта (звезда / галактика / квазар): как три популяции расходятся ещё до моделирования, почему одна фича делает почти всю работу, и как валидировать без утечки.

Что внутри:
- **color-color разделение** классов (u−g vs g−r) — видно до всякой модели;
- **«ловушка высокого MI»** — фича с высоким mutual information, но нулевым приростом (наглядный урок новичку);
- **leak-free стратифицированная CV**, confusion на границе GALAXY↔QSO, чистый LightGBM-baseline.

**Оригинал с выводами:** https://www.kaggle.com/code/georgymamarin/an-almost-exhaustive-stellar-class-eda

<p align="center">
  <img src="color_color.png" width="480" alt="color-color разделение классов"><br>
  <img src="redshift_by_class.png" width="480" alt="распределение redshift по классам">
</p>

> `stellar-class-eda.ipynb` (версия **с выводами**) добавляется отдельно — за полным рендером пока на Kaggle по ссылке выше.
