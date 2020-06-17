# You can use other dice loss functions also or weighted categorical crossentropy
# Adam is a very good optimizer but you can also use SGD
# Don't use accuracy as a metric as it gives misleading results due to the class imbalance problem, use dice coefficient, or IOU

model = UNet()
model.compile(optimizer=Adam(lr=0.0001), loss= dice_loss,  
              metrics= [dice_coef_0, dice_coef_1, dice_coef_2, dice_coef_4, dice_score])
model.summary()

print(model.metrics_names)

# Checkpoint 
# assign appropriate directory in the checkpoint filepath
filepath="Unet_checkpoints/weights-improvement-{epoch:02d}-{dice_score:.3f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor= dice_score, verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

batch_size = 32 #lower the batch size if you run out of memory
epochs = 100

model_train = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True, 
                        callbacks=callbacks_list, verbose=1, validation_data=(X_valid, Y_valid) )


# To plot the Graph of training loss vs validation loss

loss = model_train.history['loss']
val_loss = model_train.history['val_loss']
epochs = range(100)
plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
