results = model.evaluate(X_test, Y_test, batch_size=32)

print('test loss : ', results[0])
print('dice_score_0 : ', results[1])
print('dice_score_1 : ', results[2])
print('dice_score_2 : ', results[3])
print('dice_score_4 : ', results[4])
print('dice_score_mean : ', results[5])


pred = model.predict(X_test)
print(pred.shape)

predicted_images = np.argmax(pred, axis=-1)

plt.figure(figsize=[5,5])
%matplotlib inline
plt.subplot(121)
curr_img = np.reshape(predicted_images[1],(240,240))
plt.imshow(curr_img, cmap='gray')
