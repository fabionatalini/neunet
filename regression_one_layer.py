from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils.vis_utils import plot_model

file_path="/home/fabio/Documents/neunet/"

datos=load_boston(return_X_y=False)
#print(datos.feature_names)
#print(datos.DESCR)
X_data=datos.data
y_data=datos.target

def regresion(do_plot=False,figure_name="model_plot.png",print_summary=False):
    """
    When do_plot=True, it assumes that the graphviz library is installed
    sudo apt-get install graphviz
    and also the python modules graphviz and pydot are installed
    """
    model=Sequential()
    model.add(Dense(1,input_dim=13))
    #model.add(Dense(1))
    model.compile(loss="mean_squared_error",optimizer="adam")
    if do_plot:
        plot_model(model, to_file=figure_name, show_shapes=True, show_layer_names=True)
    if print_summary:
        print(model.summary())
    return model

regresion(do_plot=True,figure_name=file_path+"model_plot3.png",print_summary=False)

estimator=KerasRegressor(build_fn=regresion, epochs=50, batch_size=10)
estimator.fit(X_data, y_data, verbose=False)

weights = estimator.model.layers[0].get_weights()[0]