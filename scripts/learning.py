import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
import pandas as pd

def run_model(i):
    # Parameters
    
    data_dir = pathlib.Path('../cleaned') 
    print(data_dir)
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)

    # We have a 70/30 split here, we split the training and testing once
    train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

    class_names = train_dataset.class_names

    # We're making a test set now
    val_batches = tf.data.experimental.cardinality(val_ds) # gets number of batches and splits it into 20/80
    test_dataset = val_ds.take(val_batches // 5)
    validation_dataset = val_ds.skip(val_batches // 5)

    # Just prepares image so that they're easy to use, makes it optimal for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # augmenting data
    data_augmentation = tf.keras.Sequential([
       tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ])


    # This is essentially what the augmented data would look like


    # images are currently in a (0,255) pixel format, so we transform for transfer learning
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    # Create base model based off of Google's Mobile Net, we will use the 'bottleneck' layer for feature extraction, 
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape) # 5 by 5 and 1280 features

    base_model.trainable = False
    base_model.summary()

    # transforms image / bottleneck layer into a format that we can make predictions for.
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    # make predicition layer
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Maybe we could change the dropout rate?
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)


    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'), tf.keras.metrics.Precision(name='precision')])

    initial_epochs = 13
    loss0, accuracy0, precision0 = model.evaluate(validation_dataset) # calculate loss and accuracy
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))
    print("initial precision: {:.2f}".format(precision0))

    history = model.fit(train_dataset,
                        epochs=initial_epochs,
                        validation_data=validation_dataset) # accuracy on ten epochs, ours is pretty bad right now

    # charts showing accuracy and loss
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    precision = history.history['precision']
    val_precision = history.history['val_precision']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # To keep the best model, save the model every time, use the same parameters
    # If you want to avoid overfitting, in many cases dropping doesn't make a big difference. Those two numbers
    # don't make 

    # Try different epochs and different results. See if there's any patterns, around this range of batch sizes we get consistently
    # better results. 

    # Sometimes you can try twenty vs eighty,

    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards, how do I decide whether to improve this?
    fine_tune_at = 70

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    '''lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.9)'''

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
                metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'), tf.keras.metrics.Precision(name='precision')])

    model.summary()

    fine_tune_epochs = 13
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(train_dataset,
                            epochs=total_epochs,
                            initial_epoch=len(history.epoch),
                            validation_data=validation_dataset)

    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']

    precision += history.history['precision']
    val_precision += history.history['val_precision']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.3, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    loss, accuracy, prec = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    model.save('../models/eight/' + f"model{i}.keras")

    return loss, accuracy, prec 
 


basic_df = pd.DataFrame({'Model': [], 'Accuracy': [], 'Loss': [], 'Precision': []})
for i in range(10):
    loss, acc, prec = run_model(i)
    basic_df.loc[len(basic_df)] = [f'Model {i}', acc, loss, prec]

print(basic_df)