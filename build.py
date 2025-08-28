from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import InputLayer
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

def apply_structural_damage(model):
    """
    Apply structural damage to a random Dense layer with a random mode.

    Modes:
        - 'zero': set weights to 0
        - 'random': set weights to small random values

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
    mode = random.choice(['zero','random'])

    weights = damaged_layer.get_weights()
    if not weights:
        return model, layer_name, mode  # no weights to damage

    if mode == 'zero':
        damaged_weights = [np.zeros_like(w) for w in weights]
    elif mode == 'random':
        damaged_weights = [np.random.randn(*w.shape) * 0.1 for w in weights]

    damaged_layer.set_weights(damaged_weights)
    return model, layer_name, mode, weights,damaged_weights


def create_base_model(a,b,hidden,act='relu',out_act='softmax'):
    inp=Input(shape=(a,))
    x=inp
    for i,u in enumerate(hidden): x=Dense(u,activation=act,name=f'dense{i}')(x)
    out=Dense(b,activation=out_act,name='output')(x)
    return Model(inp,out)

def get_acc(model, X, y_true):
    loss, acc = model.evaluate(X, y_true, verbose=0)
    return round(acc * 100, 2)





def get_layer_outputs(model, X):
    outputs = []
    x = X
    layer_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

    for lname in layer_names:
        layer = model.get_layer(lname)
        x = layer(x)  # feed current data to next layer
        outputs.append(x.numpy())  # convert Tensor to numpy for consistency

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
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False
    layer_names = [l.name for l in model.layers]
    idx = layer_names.index(damaged_layer)
    # Determine patch input/output dims
    if damaged_layer == 'dense0':
        patch_input_dim = input_shape[0]
    else:
        patch_input_dim = model.get_layer(layer_names[idx - 1]).units

    if damaged_layer == 'output':
        patch_output_dim = y_train.shape[1]
        patch_activation = 'softmax'
        patch_loss = 'categorical_crossentropy'
    else:
        patch_output_dim = model.get_layer(damaged_layer).units
        patch_activation = 'relu'
        patch_loss = 'mse'
    # Create patch layer
    patch_layer = Dense(patch_output_dim, activation=patch_activation, name='patch')
    patch_model = Sequential([patch_layer], name='patch')
    patch_model.compile(optimizer='adam', loss=patch_loss)

    x = inputs = Input(shape=model.input_shape[1:])
    for i in range(idx):
        if isinstance(model.layers[i], InputLayer):
            continue
        x = model.layers[i](x)
    x = patch_layer(x)
    for i in range(idx + 1, len(model.layers)):
        if isinstance(model.layers[i], InputLayer):
            continue
        x = model.layers[i](x)


    healing_model = Model(inputs, outputs=x)
    healing_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train only patch
    history = healing_model.fit(
        x_train[:train_samples], y_train[:train_samples],
        epochs=epochs, batch_size=batch_size
    )

    return healing_model, patch_model, history, patch_layer.get_weights()

#-----------------------------------------------------------------------------------


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

def get_damaged_layer(model,X_test,y_test_cat,attack_type):
    X_clean = X_test[:100]
    n=100
    if attack_type == 'FGSM':
        X_adv = fgsm_attack(model, X_test[:n], y_test_cat[:n])
    elif attack_type == 'PGD':
        X_adv = pgd_attack(model, X_test[:n], y_test_cat[:n])

    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Get activations
    clean_activations = activation_model.predict(X_clean)
    adv_activations = activation_model.predict(X_adv)

    layer_differences = []
    for clean, adv in zip(clean_activations, adv_activations):
        mse = np.mean((clean - adv) ** 2)
        layer_differences.append(mse)

    return model.layers[np.argmax(layer_differences)].name,layer_differences,[layer.name for layer in model.layers]

def build_patch(output_dim):
    return Sequential([
        Dense(output_dim, activation='relu',name='patch_0'),
        Dense(output_dim, activation='linear',name='patch_1')
    ],name="patch")


def integrate_patch(model, damaged_layer, patch):
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index(damaged_layer)

    x = inputs = model.input

    for i in range(layer_idx):
        if isinstance(model.layers[i], InputLayer):
            continue
        x = model.layers[i](x)

    x = patch(x)

    for i in range(layer_idx + 1, len(model.layers)):
        if isinstance(model.layers[i], InputLayer):
            continue
        x = model.layers[i](x)

    healed_model = Model(inputs=inputs, outputs=x)
    return healed_model



def freeze_except_patch(healed_model, patch):

    for layer in healed_model.layers:
        layer.trainable = False
    for layer in patch.layers:
        layer.trainable = True
    
def prepare_adversarial_training_data(model, X_train, y_train_cat, attack_type, n=15000):
    # Select the attack type
    if attack_type == 'FGSM':
        X_adv = fgsm_attack(model, X_train[:n], y_train_cat[:n])
    elif attack_type == 'PGD':
        X_adv = pgd_attack(model, X_train[:n], y_train_cat[:n])

    # Combine clean and adversarial data
    y_adv = y_train_cat[:n]
    X_total = np.concatenate([X_train[:n], X_adv])
    y_total = np.concatenate([y_train_cat[:n], y_adv])
    return X_total, y_total

def train_healed_model(healed_model,model, X_train, y_train_cat,attack_type):


    healed_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    X_total, y_total = prepare_adversarial_training_data(model, X_train, y_train_cat,attack_type)
    healed_model.fit(X_total, y_total, epochs=10, batch_size=128, validation_split=0.1)
    return healed_model

#--------------------------------------------------------------------------------------------------------------------------

def show_layer_damage_circles(layer_differences, layer_names, damaged_layer_name,filename, st):
    st.markdown("## 🧠 Neural Network Layer Damage Map")

    st.info(
        "Each circle represents a layer in the neural network.\n\n"
        "🔵 The number inside shows how much that layer changed when attacked.\n\n"
        "🔴 The red circle is the most damaged layer — where the model was hurt most by the adversarial attack."
    )

    num_layers = len(layer_differences)
    x_positions = np.linspace(1, num_layers, num_layers)
    y_position = 1

    fig, ax = plt.subplots(figsize=(num_layers * 1.2, 3))  
    for i in range(num_layers):
        x = x_positions[i]
        mse = layer_differences[i]
        label = layer_names[i]
        color = 'red' if label == damaged_layer_name else 'skyblue'

        circle = plt.Circle((x, y_position), 0.4, color=color, ec='black', lw=1.5)
        ax.add_patch(circle)

        ax.text(x, y_position, f"{mse:.1e}", fontsize=8.5, ha='center', va='center', color='black')

        ax.text(x, y_position - 0.6, label, fontsize=9, ha='center', va='center', rotation=0)

    damaged_idx = layer_names.index(damaged_layer_name)
    x_arrow = x_positions[damaged_idx]
    ax.annotate('Damaged Layer',
                xy=(x_arrow, y_position + 0.5),
                xytext=(x_arrow, y_position + 1.2),
                ha='center', fontsize=10, color='red',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))

    ax.set_xlim(0, num_layers + 1)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig("images/"+filename, dpi=500, bbox_inches="tight")

    st.pyplot(fig)


def show_layer_damage_circles_for_struc(layer_differences, layer_names, damaged_layer_name,filename, st):
    st.markdown("## 🧠 Neural Network Layer Damage Map")

    st.info(
        "These numbers show how much each layer’s output changed due to damage.\n\n"
        "We measure this using a **difference score** — higher means more disruption.\n\n"
        "🔍 But we don’t just pick the highest score! Damage in early layers can make later ones look worse, even if they’re fine."
    )

    num_layers = len(layer_differences)
    x_positions = np.linspace(1, num_layers, num_layers)
    y_position = 1

    fig, ax = plt.subplots(figsize=(num_layers * 1.2, 3))  
    for i in range(num_layers):
        x = x_positions[i]
        mse = layer_differences[i]
        label = layer_names[i]
        color = 'red' if label == damaged_layer_name else 'skyblue'

        circle = plt.Circle((x, y_position), 0.4, color=color, ec='black', lw=1.5)
        ax.add_patch(circle)

        ax.text(x, y_position, f"{mse:.1e}", fontsize=8.5, ha='center', va='center', color='black')

        ax.text(x, y_position - 0.6, label, fontsize=9, ha='center', va='center', rotation=0)

    damaged_idx = layer_names.index(damaged_layer_name)
    x_arrow = x_positions[damaged_idx]
    ax.annotate('Damaged Layer',
                xy=(x_arrow, y_position + 0.5),
                xytext=(x_arrow, y_position + 1.2),
                ha='center', fontsize=10, color='red',
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))

    ax.set_xlim(0, num_layers + 1)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig("images/"+filename, dpi=500, bbox_inches="tight")
    st.pyplot(fig)


def show_layer_patch_circles_for_struc(layer_differences, layer_names, patch_layer_name,filename, st):
    st.markdown("## 🧠 Neural Network Layer Damage Map")

    st.info(
        "These numbers show how much each layer’s output changed due to damage.\n\n"
        "We measure this using a **difference score** — higher means more disruption.\n\n"
        "🛠️ The green circle shows the **patched layer** — a trained replacement for the one we detected as damaged."
    )

    num_layers = len(layer_differences)
    x_positions = np.linspace(1, num_layers, num_layers)
    y_position = 1

    fig, ax = plt.subplots(figsize=(num_layers * 1.2, 3))
    for i in range(num_layers):
        x = x_positions[i]
        mse = layer_differences[i]
        label = layer_names[i]
        color = 'green' if label == patch_layer_name else 'skyblue'

        circle = plt.Circle((x, y_position), 0.4, color=color, ec='black', lw=1.5)
        ax.add_patch(circle)

        ax.text(x, y_position, f"{mse:.1e}", fontsize=8.5, ha='center', va='center', color='black')
        ax.text(x, y_position - 0.6, label, fontsize=9, ha='center', va='center', rotation=0)

    patch_idx = layer_names.index(patch_layer_name)
    x_arrow = x_positions[patch_idx]
    ax.annotate('Patched Layer',
                xy=(x_arrow, y_position + 0.5),
                xytext=(x_arrow, y_position + 1.2),
                ha='center', fontsize=10, color='green',
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6))

    ax.set_xlim(0, num_layers + 1)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig("images/"+filename, dpi=500, bbox_inches="tight")
    st.pyplot(fig)



def show_patch_layer_replacement(layer_names, layer_differences, patched_layer_name,filename, st):
    st.markdown("## 🧩 Patched Neural Network Layer Map")

    st.success(
        "We've repaired the most damaged layer! 🔧\n\n"
        "Each circle below shows a layer in the model.\n\n"
        "🟢 The patched layer is highlighted — it's a fresh new layer that replaces the damaged one!\n\n"
        "Layers labeled **Frozen** are locked (not updated), while the patched layer is still trainable.\n\n"
        "🔢 The number inside each circle shows how much that layer changed during attack (MSE)."
    )

    num_layers = len(layer_names)
    x_positions = np.linspace(1, num_layers, num_layers)
    y_position = 1

    fig, ax = plt.subplots(figsize=(num_layers * 1.2, 3))

    for i in range(num_layers):
        x = x_positions[i]
        label = layer_names[i]
        mse = layer_differences[i]  # Assuming this corresponds to each layer
        is_patch = (label == patched_layer_name)
        color = 'limegreen' if is_patch else 'lightgray'

        # Draw the layer as a circle
        circle = plt.Circle((x, y_position), 0.4, color=color, ec='black', lw=1.5)
        ax.add_patch(circle)

        # Display the number (e.g., MSE) inside the circle
        ax.text(x, y_position, f"{mse:.1e}", fontsize=8, ha='center', va='center', color='black')

        # Layer name below the circle
        ax.text(x, y_position - 0.6, label, fontsize=7, ha='center', va='center', color='black')

        # Trainable status
        train_text = "Trainable" if is_patch else "Frozen"
        train_color = 'green' if is_patch else 'gray'
        ax.text(x, y_position - 0.9, train_text, fontsize=8, ha='center', va='center', color=train_color)

    # Arrow pointing to the patched layer
    patched_idx = layer_names.index(patched_layer_name)
    x_arrow = x_positions[patched_idx]
    ax.annotate('Patched Layer',
                xy=(x_arrow, y_position + 0.5),
                xytext=(x_arrow, y_position + 1.2),
                ha='center', fontsize=10, color='green',
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6))

    ax.set_xlim(0, num_layers + 1)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig("images/"+filename, dpi=500, bbox_inches="tight")
    st.pyplot(fig)



def show_patch_layer_replacement_struc(layer_names, patched_layer_name,filename, st):
    st.markdown("## 🧩 Patched Neural Network Layer Map")

    st.success(
        "We've repaired the most damaged layer! 🔧\n\n"
        "Each circle below shows a layer in the model.\n\n"
        "🟢 The patched layer is highlighted — it's a fresh new layer that replaces the damaged one!\n\n"
        "Layers labeled **Frozen** are locked (not updated), while the patched layer is still trainable."
    )

    num_layers = len(layer_names)
    x_positions = np.linspace(1, num_layers, num_layers)
    y_position = 1

    fig, ax = plt.subplots(figsize=(num_layers * 1.2, 3))

    for i in range(num_layers):
        x = x_positions[i]
        label = layer_names[i]
        is_patch = (label == patched_layer_name)
        color = 'limegreen' if is_patch else 'lightgray'

        # Draw the layer as a circle
        circle = plt.Circle((x, y_position), 0.4, color=color, ec='black', lw=1.5)
        ax.add_patch(circle)

        # Layer name inside the circle
        ax.text(x, y_position, label, fontsize=7, ha='center', va='center', color='black')

        # Trainable status below the circle
        train_text = "Trainable" if is_patch else "Frozen"
        train_color = 'green' if is_patch else 'gray'
        ax.text(x, y_position - 0.6, train_text, fontsize=8, ha='center', va='center', color=train_color)

    # Arrow to patched layer
    patched_idx = layer_names.index(patched_layer_name)
    x_arrow = x_positions[patched_idx]
    ax.annotate('Patched Layer',
                xy=(x_arrow, y_position + 0.5),
                xytext=(x_arrow, y_position + 1.2),
                ha='center', fontsize=10, color='green',
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6))

    ax.set_xlim(0, num_layers + 1)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig("images/"+filename, dpi=500, bbox_inches="tight")
    st.pyplot(fig)
