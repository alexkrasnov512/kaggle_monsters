# Ghouls, Goblins, and Ghosts... Boo!
В данном проекте представлено решение соревнования на Kaggle, в котором необходимо классифицировать различных монстров. Соревнование можно найти по ссылке: https://www.kaggle.com/competitions/ghouls-goblins-and-ghosts-boo/overview.
Нам был предоставлен обучающий и тестовый файл. В обучающем файле содержится информация о 371 существах, которых удалось идентифицировать. 
Чтобы опознать остальных, нужно сопоставить признаки для каждого из существ воспользовавшись обучающей выборкой и предсказать какие существа находятся в тестовой выборке.
Для решения данной задачи я воспользовался различным методами классификации(метод k-ближайших соседей, метод опорных векторов, деревья решений, логистическая регрессия и наивный Байес). 
Также на данном наборе были опробованы ансамблевые методы, такие как стекинг, беггинг, бустинг (Adaboost и Gradientboost), а также экстремальный бустинг XGBoost. 
Кроме того использовалась нейронная сеть с 1 скрытым слоем с сигмоидной функцией активации. 
Вывод - так как количество существ в обучающем файле меньше, чем в тестовом, точность определения существ не была выше 73% для различных методов классификации. 
Так, например, для нейронной сети точность определения была 72.2%, для SVC - 72.4%, для логистической регрессии - 73 %, у ансамблевых методов лучший результат был у стекинга - 72.4%, у беггинга - 73.5 %.
Следовательно для данного соревнования можно было воспользоваться простыми методами классификации (логистической регрессией или методом опорных векторов).