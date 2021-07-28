from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import numpy
import matplotlib.pyplot as plt
#import pprint
import pandas

#read in data
datos=load_boston(return_X_y=False)
#print(datos.feature_names)
#print(datos.DESCR)
X_data=datos.data
y_data=datos.target

def viewer(predictions,actual,ccoef):
    fig,ax=plt.subplots()
    ax.plot(predictions,color="b",label="pronostico",linewidth=1)
    ax.plot(actual,color="r",label="real",linewidth=1)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.legend(loc='lower left')
    plt.title("Correl. coeff. {a}".format(a=ccoef))
    plt.show()

def NNregressor(do_plot=False,figure_name="model_plot.png",print_summary=False):
    """
    When do_plot=True, it assumes that the graphviz library is installed;
    in Linux this can be done as follows:
    sudo apt-get install graphviz
    Moreover, it is assumed that the python modules graphviz and pydot are installed in the environment
    """
    model=Sequential()
    model.add(Dense(1,input_dim=13, activation="linear"))
    model.compile(loss="mean_squared_error",optimizer="adam")
    if do_plot:
        plot_model(model, to_file=figure_name, show_shapes=True, show_layer_names=True)
    if print_summary:
        print(model.summary())
    return model

#NNregressor(do_plot=True,figure_name="/home/fabio/Documents/neunet/model_plot3.png",print_summary=False)
estimator=NNregressor()
estimator.fit(x=X_data, y=y_data, epochs=500, verbose=False)
pronostico=estimator.predict(x=X_data)
cc=round(numpy.corrcoef(x=y_data, y=pronostico[:,0])[0,1],3)
viewer(predictions=pronostico,actual=y_data,ccoef=cc)

#pprint.pprint(estimator.get_config())
NNw1 = estimator.get_weights()[0][:,0]


########################## simple regression model with scikit-learn #####################
from sklearn.linear_model import LinearRegression

def sklearnLM(xs,ys):
    lrmodel=LinearRegression()
    lrmodel.fit(X=xs,y=ys)
    predi=lrmodel.predict(X=xs)
    c=round(numpy.corrcoef(x=ys, y=predi)[0,1],3)
    viewer(predictions=predi,actual=ys,ccoef=c)
    return lrmodel

reg=sklearnLM(xs=X_data,ys=y_data)
reg_coef = reg.coef_


########################## compare regression coefficients with NN weights #####################
pandas.DataFrame({"nn":NNw1,"reg":reg_coef})

fig,ax=plt.subplots()
#ax.set_ylim(bottom=-5, top=5)
ax.set_ylim(bottom=-2, top=2)
positions=numpy.arange(len(NNw1))
ax.bar(positions+0.00, height=NNw1, color = 'red', width = 0.5)
ax.bar(positions+0.5, height=reg_coef, color = 'blue', width = 0.5)
plt.show()












