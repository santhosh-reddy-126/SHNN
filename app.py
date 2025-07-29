import streamlit as st
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns

from build import (
    create_base_model, get_acc,
    apply_structural_damage,
    get_layer_outputs,
    compare_saved_outputs,
    find_damaged_layer,
    train_healing_patch,
    fgsm_attack, pgd_attack, generate_adversarial_examples,
    find_damaged_layer,
    train_healed_model
)

st.set_page_config(page_title="Simulate SHNN", layout="wide")
st.title("🧠 Simulate Self-Healing Neural Network")

uploaded_file = st.file_uploader("📂 Upload MNIST .mat file (e.g., mnist-original.mat)", type=["mat"])

def plot_bar(metrics, title):
    fig, ax = plt.subplots()
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis", ax=ax)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    st.pyplot(fig)

def get_layer_outputs(model, X, layer_names=['dense0', 'dense1', 'dense2', 'output']):
    outputs = []
    for lname in layer_names:
        layer = model.get_layer(lname)
        intermediate_model = Model(inputs=model.input, outputs=layer.output)
        outputs.append(intermediate_model.predict(X))
    return outputs

def compare_saved_outputs(ref_outputs, new_outputs):
    return [np.mean(np.abs(ref - new)) for ref, new in zip(ref_outputs, new_outputs)]

if uploaded_file:
    st.success("✅ File uploaded!")

    try:
        if 'data' not in st.session_state:
            mnist = loadmat(uploaded_file)
            mnist_data = mnist["data"].T
            mnist_label = mnist["label"][0]
            data = mnist_data / 255.0

            X_train, X_test, y_train, y_test = train_test_split(data, mnist_label, test_size=0.2, random_state=42)
            y_train_cat = to_categorical(y_train)
            y_test_cat = to_categorical(y_test)

            st.session_state.update({
                "X_train": X_train,
                "X_test": X_test,
                "y_train_cat": y_train_cat,
                "y_test_cat": y_test_cat,
                "data": data,
                "y": mnist_label,
                "model_created": False
            })

        st.subheader("📊 Dataset Overview")
        st.write(f"Data shape: `{st.session_state.data.shape}`")
        st.write(f"Label shape: `{st.session_state.y.shape}`")
        st.write(f"Unique classes: {np.unique(st.session_state.y)}")

        if st.button("🚀 Create Base Model"):
            model = create_base_model()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            with st.spinner("Training base model (2 epochs)..."):
                model.fit(st.session_state.X_train, st.session_state.y_train_cat, epochs=2, batch_size=128, verbose=0)
                loss, acc = model.evaluate(st.session_state.X_test, st.session_state.y_test_cat, verbose=0)
            st.session_state.update({
                "model": model,
                "model_created": True,
                "ref_outputs": get_layer_outputs(model, st.session_state.X_test[:10]),
                "X_sample_ref": st.session_state.X_test[:10]
            })
            st.success(f"✅ Base Model Trained - Accuracy: **{acc:.2%}**")

        if st.session_state.get("model_created"):
            st.header("⚔️ Choose Repair Path")
            path = st.radio("Select repair path:", ["Adversarial", "Structural"], horizontal=True)

            # ----------------------------- Adversarial Path -----------------------------
            if path == "Adversarial":
                st.subheader("🔐 Adversarial Repair")
                model = st.session_state.model
                X_sample = st.session_state.X_test[:100]
                y_sample = st.session_state.y_test_cat[:100]
                attack_type = st.selectbox("⚔️ Choose Adversarial Attack Type", ["FGSM", "PGD", "General"])
                if attack_type == "FGSM":
                    X_adv = fgsm_attack(model, X_sample, y_sample)
                elif attack_type == "PGD":
                    X_adv = pgd_attack(model, X_sample, y_sample)
                elif attack_type == "General":
                    X_adv = generate_adversarial_examples(model, X_sample, y_sample)

                if st.button("🔒 Evaluate Model on Adversarial Samples"):

                    preds = {
                        "Normal": model.predict(X_sample),
                        attack_type: model.predict(X_adv)
                    }

                    accs = {k: np.mean(np.argmax(v, axis=1) == np.argmax(y_sample, axis=1)) * 100 for k, v in preds.items()}

                    st.session_state.update({
                        "X_sample": X_sample,
                        "y_sample": y_sample,
                        "X_adv": X_adv,
                        "attack_type": attack_type
                    })

                    plot_bar(accs, "Accuracy Before Adversarial Healing")

                if st.button("🛡️ Heal Model with Adversarial Training"):
                    with st.spinner("Healing using adversarial examples..."):
                        model = st.session_state.model
                        X_train = st.session_state.X_train
                        y_train_cat = st.session_state.y_train_cat
                        X_test = st.session_state.X_test
                        y_test_cat = st.session_state.y_test_cat
                        healed_model = train_healed_model(model, None, X_train, y_train_cat, X_test,y_test_cat,attack_type)
                        st.session_state.model = healed_model
                        st.success("✅ Model healed using adversarial training")

                if st.button("📈 Re-evaluate After Adversarial Repair"):
                    model = st.session_state.model
                    X_sample = st.session_state.X_sample
                    y_sample = st.session_state.y_sample
                    preds = {
                        "Normal": model.predict(X_sample),
                        attack_type: model.predict(st.session_state.X_adv)
                    }
                    accs = {k: np.mean(np.argmax(v, axis=1) == np.argmax(y_sample, axis=1)) * 100 for k, v in preds.items()}
                    plot_bar(accs, "Accuracy After Adversarial Healing")

            # ----------------------------- Structural Path -----------------------------
            elif path == "Structural":
                st.subheader("🧱 Structural Damage Repair")

                if st.button("💥 Apply Random Structural Damage"):
                    st.session_state.model, layer, mode = apply_structural_damage(st.session_state.model)
                    st.session_state.damaged_layer = layer
                    st.error(f"💔 Structural damage applied to `{layer}` using `{mode}`")

                if st.button("🩻 Detect Damaged Layers"):
                    curr_outputs = get_layer_outputs(st.session_state.model, st.session_state.X_test[:10])
                    diffs = compare_saved_outputs(st.session_state.ref_outputs, curr_outputs)
                    layer_names = ['dense0', 'dense1', 'dense2', 'output']
                    layer = find_damaged_layer(diffs, layer_names)
                    st.session_state.damaged_layer = layer
                    st.warning(f"🔍 Most likely damaged layer: `{layer}`")

                if st.button("📉 Test Accuracy After Damage"):
                    acc = get_acc(st.session_state.model, st.session_state.X_test, st.session_state.y_test_cat)
                    st.metric("🎯 Accuracy After Damage", f"{acc:.2f}%")
                    st.progress(acc / 100)

                if st.button("🩹 Heal the Model"):
                    with st.spinner("Healing structurally damaged model..."):
                        healed_model, _, history = train_healing_patch(
                            st.session_state.model,
                            st.session_state.damaged_layer,
                            st.session_state.X_train,
                            st.session_state.y_train_cat
                        )
                        st.session_state.model = healed_model
                        st.success("✅ Model healing complete")
                        fig, ax = plt.subplots()
                        ax.plot(history.history['accuracy'], label='Training Accuracy')
                        ax.set_title("📈 Patch Training Accuracy (Structural Healing)")
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Accuracy")
                        ax.legend()
                        st.pyplot(fig)

                if st.button("📊 Calculate Test Accuracy"):
                    acc = get_acc(st.session_state.model, st.session_state.X_test, st.session_state.y_test_cat)
                    st.metric("🎯 Final Test Accuracy", f"{acc:.2f}%")
                    st.progress(acc / 100)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

