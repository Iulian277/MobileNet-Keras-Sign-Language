from imports import *
from data_processing import train_batches, valid_batches

# Download (/ Load) the MobileNet model
mobile = tf.keras.applications.mobilenet.MobileNet()

# Exclude the last 5 layers from MobileNet model
x = mobile.layers[-6].output

# Add *before* the dense output layer with 10 classes the layers stored in 10
output = Dense(units = 10, activation = 'softmax')(x)

# Create the model
model = Model(inputs = mobile.input, outputs = output)

# Freeze some layers (the base)
# Train only the last 23 layers
for layer in model.layers[:-23]:
    layer.trainable = False

# Train the model
model.compile(optimizer = Adam(learning_rate = 0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x = train_batches,
          validation_data = valid_batches,
          epochs = 30,
          verbose = 2)

# Save the model
if os.path.isfile('models/mobile_net_sign_language_model.h5') is False:
    model.save('models/mobile_net_sign_language_model.h5')