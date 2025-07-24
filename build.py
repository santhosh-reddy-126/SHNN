from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf
import numpy as np
import random

def apply_structural_damage(model):
    """
    Apply structural damage to a random Dense layer with a random mode.

    Modes:
        - 'zero': set weights to 0
        - 'random': set weights to small random values
        - 'scale': scale weights down by 0.01

    Returns:
        model: Damaged model
        layer_name: Damaged layer name
        mode: Damage mode used
    """
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if not dense_layers:
        raise ValueError("No Dense layers found in the model to damage.")

    damaged_layer = random.choice(dense_layers)
    layer_name = damaged_layer.name
    mode = random.choice(['zero', 'random', 'scale'])

    weights = damaged_layer.get_weights()
    if not weights:
        return model, layer_name, mode  # no weights to damage

    if mode == 'zero':
        damaged_weights = [np.zeros_like(w) for w in weights]
    elif mode == 'random':
        damaged_weights = [np.random.randn(*w.shape) * 0.1 for w in weights]
    elif mode == 'scale':
        damaged_weights = [w * 0.01 for w in weights]

    damaged_layer.set_weights(damaged_weights)
    return model, layer_name, mode


def create_base_model():
    inputs = Input(shape=(784,))
    x = Dense(256, activation='relu', name='dense0')(inputs)
    x = Dense(128, activation='relu', name='dense1')(x)
    x = Dense(64, activation='relu', name='dense2')(x)
    outputs = Dense(10, activation='softmax', name='output')(x)
    return Model(inputs, outputs)

def get_acc(model, X, y_true):
    loss, acc = model.evaluate(X, y_true, verbose=0)
    return round(acc * 100, 2)


layer_names = ['dense0', 'dense1', 'dense2', 'output']

def get_layer_outputs(model, X, layer_names=layer_names):
    outputs = []
    for lname in layer_names:
        layer = model.get_layer(lname)
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        out = intermediate_model.predict(X, verbose=0)
        outputs.append(out)
    return outputs


def compare_saved_outputs(ref_outputs, new_outputs):
    diffs = []
    for ref, new in zip(ref_outputs, new_outputs):
        diff = np.mean(np.abs(ref - new))
        diffs.append(diff)
    return diffs

def find_damaged_layer(diffs, layer_names, threshold=0.1):
    for i in range(len(diffs)):
        if diffs[i] > threshold:
            if i == 0 or (diffs[i] - diffs[i-1]) > threshold:
                return layer_names[i]
    return layer_names[np.argmax(diffs)]


def train_healing_patch(model, damaged_layer, x_train, y_train,
                        input_shape=(784,), train_samples=10000,
                        epochs=5, batch_size=128):
    for layer in model.layers:
        layer.trainable = False
    layer_names = [l.name for l in model.layers]
    idx = layer_names.index(damaged_layer)

    # Determine patch input dim
    if damaged_layer == 'dense0':
        patch_input_dim = input_shape[0]
    else:
        patch_input_dim = model.get_layer(layer_names[idx - 1]).units

    # Determine patch output dim and loss
    if damaged_layer == 'output':
        patch_output_dim = y_train.shape[1]
        patch_activation = 'softmax'
        patch_loss = 'categorical_crossentropy'
    else:
        patch_output_dim = model.get_layer(damaged_layer).units
        patch_activation = 'relu'
        patch_loss = 'mse'

    # Define patch
    patch = Sequential([
        Dense(patch_output_dim, activation=patch_activation, input_shape=(patch_input_dim,))
    ])
    patch.compile(optimizer='adam', loss=patch_loss)

    # Freeze original model layers
    for layer in model.layers:
        layer.trainable = False

    inputs = Input(shape=input_shape)

    # Model part before damaged layer
    if damaged_layer == 'dense0':
        x = inputs
    else:
        prev_layer_name = layer_names[idx - 1]
        prev_model = Model(inputs=model.input, outputs=model.get_layer(prev_layer_name).output)
        x = prev_model(inputs)

    # Patch replaces damaged layer
    x = patch(x)

    # Model part after damaged layer
    if damaged_layer == layer_names[-1]:  # damaged layer is last layer
        outputs = x
    else:
        next_layer_name = layer_names[idx + 1]
        # Build model from patch output to final output
        post_model = Model(inputs=model.get_layer(next_layer_name).input, outputs=model.output)
        outputs = post_model(x)

    healing_model = Model(inputs, outputs)
    healing_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    healing_model.fit(x_train[:train_samples], y_train[:train_samples], epochs=epochs, batch_size=batch_size)

    return healing_model, patch


def fgsm_attack(model, x, y, epsilon=0.15):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss = tf.keras.losses.categorical_crossentropy(y, prediction)
    gradient = tape.gradient(loss, x)
    signed_grad = tf.sign(gradient)
    x_adv = x + epsilon * signed_grad
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv.numpy()


def generate_adversarial_examples(model, x, y, epsilon=0.2):
    x_adv = tf.convert_to_tensor(x)
    y_true = tf.convert_to_tensor(y)

    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        pred = model(x_adv)
        loss = tf.keras.losses.categorical_crossentropy(y_true, pred)

    gradient = tape.gradient(loss, x_adv)
    signed_grad = tf.sign(gradient)
    x_adversarial = x_adv + epsilon * signed_grad
    x_adversarial = tf.clip_by_value(x_adversarial, 0, 1)
    return x_adversarial.numpy()


def pgd_attack(model, x, y, epsilon=0.2, alpha=0.01, num_iter=30):
    x_adv = tf.identity(x)
    for i in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            prediction = model(x_adv)
            loss = tf.keras.losses.categorical_crossentropy(y, prediction)
        gradient = tape.gradient(loss, x_adv)
        signed_grad = tf.sign(gradient)
        x_adv = x_adv + alpha * signed_grad
        x_adv = tf.clip_by_value(x_adv, x - epsilon, x + epsilon)  # project perturbation
        x_adv = tf.clip_by_value(x_adv, 0, 1)  # keep valid pixel range
    return x_adv.numpy()