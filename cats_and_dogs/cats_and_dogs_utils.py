import requests
import io

import keras
import sklearn.metrics
import matplotlib.pyplot
import numpy
import PIL
import PIL.ImageOps
import ipywebrtc
import vis.visualization  # keras-vis
import vis.utils


def construct_model():
    model = keras.models.Sequential()

    model.add(
        keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            input_shape=(150, 150, 3)
        )
    )
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(lr=1e-4),
        metrics=['acc']
    )

    return model


def process_image(original_image):
    """Converts a url, a local file, a captured image or a numpy array to
    the proper format for model inference.
    """
    if type(original_image) == str:
        # url
        if 'http' in original_image:
            image = PIL.Image.open(
                io.BytesIO(requests.get(original_image).content)
            )
        # local image path
        else:
            im = PIL.Image.open(original_image)
    elif type(original_image) == numpy.ndarray:
        # already a numpy array
        image = PIL.Image.fromarray((original_image * 255).astype('uint8'))
    else:
        # captured image (ipywidgets.widgets.widget_media.Image)
        image = PIL.Image.open(io.BytesIO(original_image.value))
    # cut to square
    image = PIL.ImageOps.fit(
        image, (min(image.size), min(image.size)), PIL.Image.ANTIALIAS
    )
    # resize to target size
    image.thumbnail((150, 150), PIL.Image.ANTIALIAS)
    # convert to numpy array, drop alpha channel if present
    return numpy.array(image)[:, :, :3] / 255


def predict(model, image, ax=None):
    pred = model.predict(image.reshape((1,) + image.shape))[0, 0]
    if pred > 0.5:
        title = f'{pred * 100:.0f}% dog'
    else:
        title = f'{(1 - pred) * 100:.0f}% cat'
    if ax is None:
        matplotlib.pyplot.figure()
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.axis('off')
        matplotlib.pyplot.imshow(image)
        matplotlib.pyplot.show()
    else:
        ax.set_title(title)
        ax.axis('off')
        ax.imshow(image)


def capture_image():
    recorder = ipywebrtc.ImageRecorder(
        stream=ipywebrtc.CameraStream.facing_user(audio=False)
    )
    recorder.recording = True
    recorder.autosave = False
    recorder.download()
    recorder.recording = False
    return recorder.image


def change_sign(x):
    return -x


# necessary in order to flip prediction of final layer
# for attention and activation maximization if target animal is cat
# needs to be in keras.activations
# (operator.neg doesn't work for some reason)
keras.activations.change_sign = change_sign


def modify_model(model, animal):
    model2 = model
    model2.layers[-1].activation = (
        keras.activations.change_sign if animal == 'cat'
        else keras.activations.linear
    )
    return vis.utils.utils.apply_modifications(model2)


def perfect_animal(
    model, animal, seed_input=numpy.random.rand(150, 150, 3),
    tv_weight=200, lp_norm_weight=10
):
    model2 = modify_model(model, animal)
    image = vis.visualization.visualize_activation(
        model2,
        layer_idx=-1,
        filter_indices=0,
        input_range=(0., 1.),
        seed_input=seed_input,
        tv_weight=tv_weight,
        lp_norm_weight=lp_norm_weight
    )
    return image


def attention(model, image, weight=0.5, ax=None):
    fig, axs = matplotlib.pyplot.subplots(1, 3, figsize=(14, 5))
    predict(model, image, ax=axs[0])
    for i, animal in enumerate(['cat', 'dog']):
        model2 = modify_model(model, animal)
        attention = vis.visualization.visualize_saliency(
            model2,
            layer_idx=-1,
            filter_indices=0,
            backprop_modifier='guided',
            seed_input=image
        )
        axs[i + 1].imshow(attention)
        axs[i + 1].axis('off')
        axs[i + 1].set_title(f'Attention {animal}')
    matplotlib.pyplot.show()


# functions for setup


def plot_roc(f_name, true, predicted):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true, predicted)
    auroc = sklearn.metrics.auc(fpr, tpr)
    accuracy_ratio = auroc * 2 - 1
    accuracy = ((predicted > 0.5) == (true == 1)).mean()

    matplotlib.pyplot.title(f_name)
    matplotlib.pyplot.plot(
        fpr, tpr, 'b',
        label=(
            f'AR = {(accuracy_ratio * 100):0.1f}, '
            f'AUC = {(auroc * 100):0.1f}, ACC = {(accuracy * 100):0.1f}'
        )
    )
    matplotlib.pyplot.legend(loc='lower right')
    matplotlib.pyplot.plot([0, 1], [0, 1], 'r--')
    matplotlib.pyplot.xlim([0, 1])
    matplotlib.pyplot.ylim([0, 1])
    matplotlib.pyplot.ylabel('True Positive Rate')
    matplotlib.pyplot.xlabel('False Positive Rate')
