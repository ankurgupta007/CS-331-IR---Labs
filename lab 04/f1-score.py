import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

population=[0.844, 0.732, 0.788]
cv_metrics = pd.DataFrame({"Naive-Bayes": 0.8445623342175066,
                           "Rocchio": 0.7525198938992043,
                          "KNN": 0.8015915119363395
                          },
                         index=[0])
plt.figure(figsize=(8, 6))
splot=sns.barplot(data=cv_metrics)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.4f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.xlabel("F1-scores", size=14)
plt.show()