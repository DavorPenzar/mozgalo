# Korisni materijali koji mogu pomoći u izradi projekta

## PRVI dio - Feature Engineering

### Library-ji koji analiziraju vremenske nizove
- [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/introduction.html)
- [prophet](https://github.com/facebook/prophet) --> bolja opcija od tsfresh-a

### Korisne stvari za (eksploatarnu) analizu
- [kategoricki featuri](https://www.datacamp.com/community/tutorials/categorical-data) 
- [churn rate](https://en.wikipedia.org/wiki/Churn_rate)
- [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis)

### Vremenski nizovi
- [vremenski nizovi za supervised learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
- [podjela dataset-a](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)


## DRUGI dio - Modeliranje i interpretabilnost

### Library-ji pogodni za Gradient Boosting na Decision Tree-evima
- [XGBoost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
- [catboost](https://github.com/catboost/catboost)
- [shap](https://github.com/slundberg/shap) --> ovo navodno ima neku dodatnu foru za analizu necega


*Osposobljavanje GPU-a za treniranje neuronske mreze u PyTorch-u:*
- instalirati cuda-toolkit sa [ovog linka](https://developer.nvidia.com/cuda-downloads)
- napraviti reboot
- na pocetku programa staviti liniju: 'device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')'
- pri kreiranju objekta neuronske mreze, treba dodati '.to(device)'
- u svakom dijelu programa gdje koristimo tensore koji sluze u nenakvom izracunu(recimo pri treniranju neuronske mreze
 ili evaluaciji rjesenja - npr. accuracy) treba dodati '.to(device)' na te tensore
