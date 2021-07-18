from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
import numpy

datos=load_boston(return_X_y=False)
#print(datos.feature_names)
#print(datos.DESCR)
X_data=datos.data
y_data=datos.target

def regresion():
    model=Sequential()
    model.add(Dense(13,input_dim=13,activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    return model

def viewer():
    fig,ax=plt.subplots()
    ax.plot(pronostico,color="b",label="pronostico",linewidth=1)
    ax.plot(y_data,color="r",label="real",linewidth=1)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.legend(loc='lower left')
    plt.show()


set_epochs=numpy.arange(start=50,stop=501,step=50)

coefCorrel=[]
for i in set_epochs:
    print("Training with {a} epochs...".format(a=i))
    estimator=KerasRegressor(build_fn=regresion, epochs=i, batch_size=10)
    estimator.fit(X_data, y_data, verbose=False)
    pronostico=estimator.predict(X_data)
    viewer()
    coefCorrel.append(numpy.corrcoef(x=y_data, y=pronostico)[0,1])

posiciones=[i for i in range(len(set_epochs))]
fig,ax=plt.subplots()
ax.plot(coefCorrel,color="b",linewidth=1)
ax.set_xticks(posiciones,minor=False)
ax.set_xticklabels(set_epochs)
for i,j in zip(posiciones,coefCorrel):
    ax.text(i,j,str(round(j,3)))
plt.show()

