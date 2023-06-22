from keras.callbacks import ModelCheckpoint
from keras.models import Model

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Guardar un modelo

# partiendo premisa que tienen un modelo guardado en la variable model
pathSaveModel = 'Directorio/Guardar/Modelo/'
nameModel     = 'NombreDeTuModelo'

# Esto es para guardar la arquitectura de tu modelo
model_json = model.to_json() #Aqui esta el modelo llamado 'model'
with open(pathSaveModel + nameModel + ".json", "w") as json_file:
    json_file.write(model_json)

#  Compilar el Modelo
model.compile(loss = 'binary_crossentropy' , optimizer  ='Nadam', metrics = ['accuracy'])

epochs = 20

# Definir un directorio donde quieres que se vayan guardando los pesos del mejor modelo hasta el momento
fp     = pathSaveModel + nameModel + '{epoch:02d}-{val_acc:.2f}.hdf5' # Puedes poner el nombre que quieras, pero la última parte y el .hdf5 deben ir fijas para que se guarde la info del entrenamiento (epoca y acc de validacion)
chkpt  = ModelCheckpoint(filepath=fp, save_best_only=True, save_weights_only=True)

# Entrenan el modelo y se agrega el checkpoint definido para guardar los mejores pesos 
history = model.fit(x=X_train, y=Y_train ,epochs=epochs, validation_split=0.1, callbacks = [chkpt]) # el callback = [chkpt] hará que se vayan guardando los mejores pesos

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cargar Modelo desde el archivo json creado
from keras.models import model_from_json
pathModel  = 'Directirio/Modelo/Guardado'
modelName  = 'NombreDeTuModelo'
weigths    = 'PesosDeTuModelo_15-0.97.hdf5' # Ejemplo de unos pesos guardados en la epoca 15 con 0.97 de acc_val


# Cargar archivo json
json_file = open(pathModel + modelName + '.json' , 'r')
loaded_model_json = json_file.read()
json_file.close()
 # Cargar el modelo (arquitectura)
model = model_from_json(loaded_model_json)
# Cargar los pesos del modelo
model.load_weights(pathModel + weigths)
