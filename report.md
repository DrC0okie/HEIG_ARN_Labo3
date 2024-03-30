# Practical Work 03 – Mice’s sleep stages classification with MLP


## Experiment 1

### Model summary

### Performance result

### Training history plot

### Analyse

## Experiment 2

### Model summary

### Performance result

### Training history plot

### Analyse

## Competition


### SMOTE

### AVORA

### KerasTuner

We decided to implement the KerasTuner to find the best hyperparameter for our models. 


#### First try

We edited the create_model() function to tune nearly all hyperparameters.




We tried to tune the following parameters of our model:

- Adding or not second layer
- Both layer had either 2,4 or 8 neurons
- Both layer's activation function (sigmoid or ReLu)
- The optimizer (adam or sgd)
- Learning rate (0.01,0.001, or 0.0001)
- Momentum (0, 0.8 or 0.99)
- loss function (k1_divergence, categorical cross-entropy)

Because we used the GridSearch (try every combination possible) we ended up with way too much tries
so I reduced some of the choice

Reduce both layer neurons to only 4 or 8
Remove loss function and stick with categorical cross-entropy
Remove 0 momentum
Remove 0.01 learning rate

```python

def create_model(hp):
 
  has_second_layer = hp.Boolean("has_second_layer")

  mlp = keras.Sequential()
  mlp.add(layers.Input(shape=(25,)))
  mlp.add(layers.Dense(hp.Choice("first_nb_neurons",[4,8]), activation=hp.Choice("activation",["relu","sigmoid"])))
  
  if(has_second_layer):
    mlp.add(layers.Dense(hp.Choice("second_nb_neurons",[4,8]),activation=hp.Choice("activation2nd",["relu","sigmoid"])))
           
  mlp.add(layers.Dense(3, activation="softmax"))
  
  learning_rate = hp.Choice("learning_rate", [0.001, 0.0001])
  momentum = hp.Choice("momentum", [0.8,0.99])
 
  optimizer_name = hp.Choice('optimizer', values=['adam', 'sgd'])

  if optimizer_name == 'adam':
      optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
  else :
      optimizer = keras.optimizers.SGD(learning_rate=learning_rate,momentum=momentum)
 
  mlp.compile(
      optimizer=optimizer,
      loss="categorical_crossentropy",
      metrics=[keras.metrics.F1Score(average="weighted")]
  )

  return mlp

tuner = kt.GridSearch(
    create_model,
    objective= kt.Objective("f1_score", direction="max"),
    overwrite=True,
    directory="./tuning",

)

```
After 5 hours of search, VSCode crashed because it ran out of memory. 


#### Second try

Even tho VSCode crashed, we had a good overview of some of the best hyper-parameters.

We edited KerasTuner and remove some more hyper-parameters and keep only those who we were still unsure.

We kept : 

Both layer's activation function (sigmoid or ReLu)
Both layer 8 neurons 
Optimizer: ADAM
Learning rate : 0.01
Loss function : Categorical cross-entropy


```python

def create_model(hp):

  mlp = keras.Sequential()
  mlp.add(layers.Input(shape=(25,)))
  mlp.add(layers.Dense(8, activation=hp.Choice("activation",["relu","sigmoid"])))
  mlp.add(layers.Dense(8,activation=hp.Choice("activation2nd",["relu","sigmoid"])))
           
  mlp.add(layers.Dense(3, activation="softmax")) # Three ouput
  
  learning_rate = 0.001

  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
 
  mlp.compile(
      optimizer=optimizer,
      loss="categorical_crossentropy",
      metrics=[keras.metrics.F1Score(average="weighted")]
  )

  return mlp

tuner = kt.GridSearch(
    create_model,
    objective= kt.Objective("f1_score", direction="max"),
    overwrite=True,
    directory="./tuning",
)

```

Result : Sigmoid on first layer and Relu on the second layer seems to be the best one.

Training history : 

![](./images/th_2layer_sigmo+relu.png)

Performance :

![](./images/cm_2layer_sigmo+relu.png)


Mean F1 Score across all folds: 0.885


# Third try

We weren't happy with only 0.886 of F1_score, so we tried to add more neurons/layer to see if we can get a higher score without overfitting too much.