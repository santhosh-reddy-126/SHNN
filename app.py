import streamlit as st
from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

from build import (
    create_base_model, get_acc,
    apply_structural_damage,
    get_layer_outputs,
    compare_saved_outputs,
    find_damaged_layer,
    train_healing_patch,
    fgsm_attack, pgd_attack, generate_adversarial_examples
)

st.set_page_config(page_title="Simulate SHNN", layout="wide")
st.title("📦 Upload and Visualize MNIST .mat File")

uploaded_file = st.file_uploader("Upload MNIST .mat file (e.g., mnist-original.mat)", type=["mat"])

def get_layer_outputs(model, X, layer_names=['dense0', 'dense1', 'dense2', 'output']):
    outputs = []
    for lname in layer_names:
        layer = model.get_layer(lname)
        intermediate_model = Model(inputs=model.input, outputs=layer.output)
        outputs.append(intermediate_model.predict(X))
    return outputs

def compare_saved_outputs(ref_outputs, new_outputs):
    return [np.mean(np.abs(ref - new)) for ref, new in zip(ref_outputs, new_outputs)]

if uploaded_file is not None:
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

        st.subheader("🧮 Dataset Info")
        st.write(f"Shape of data: {st.session_state.data.shape}")
        st.write(f"Shape of labels: {st.session_state.y.shape}")
        st.write(f"Unique labels: {np.unique(st.session_state.y)}")

        # --- Create and Train Model ---
        if st.button("🚀 Create Base Model"):
            model = create_base_model()
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            with st.spinner("Training for 2 epochs..."):
                model.fit(st.session_state.X_train, st.session_state.y_train_cat, epochs=2, batch_size=128, verbose=0)
                loss, acc = model.evaluate(st.session_state.X_test, st.session_state.y_test_cat, verbose=0)

            st.session_state.model = model
            st.session_state.model_created = True

            X_sample = st.session_state.X_test[:10]
            st.session_state.ref_outputs = get_layer_outputs(model, X_sample)
            st.session_state.X_sample_ref = X_sample

            st.info(f"✅ Accuracy after training: **{acc:.4f}**")

        # --- Adversarial Evaluation ---
        if st.button("🔒 Evaluate Model on Adversarial Samples"):
            if not st.session_state.model_created:
                st.warning("⚠️ Please train the model first.")
            else:
                model = st.session_state.model
                X_sample = st.session_state.X_test[:100]
                y_sample = st.session_state.y_test_cat[:100]

                try:
                    X_fgsm = fgsm_attack(model, X_sample, y_sample)
                    X_pgd = pgd_attack(model, X_sample, y_sample)
                    X_gen = generate_adversarial_examples(model, X_sample, y_sample)

                    st.session_state.update({
                        "X_fgsm": X_fgsm,
                        "X_pgd": X_pgd,
                        "X_gen": X_gen,
                        "X_sample": X_sample,
                        "y_sample": y_sample
                    })

                    preds = {
                        "Normal": model.predict(X_sample),
                        "FGSM": model.predict(X_fgsm),
                        "PGD": model.predict(X_pgd),
                        "General": model.predict(X_gen)
                    }

                    accs = {k: np.mean(np.argmax(v, axis=1) == np.argmax(y_sample, axis=1)) for k, v in preds.items()}

                    for k, v in accs.items():
                        st.metric(f"{k} Accuracy", f"{v * 100:.2f}%")

                    st.subheader("📊 Accuracy Comparison")
                    st.bar_chart({ "Accuracy (%)": {k: v * 100 for k, v in accs.items()} })

                except Exception as e:
                    st.error(f"⚠️ Failed to evaluate adversarial accuracy: {e}")

        # --- Structural Damage ---
        if st.button("💥 Apply Random Structural Damage"):
            if not st.session_state.model_created:
                st.warning("⚠️ Train the model first.")
            else:
                st.session_state.model, layer, mode = apply_structural_damage(st.session_state.model)
                st.session_state.damaged_layer = layer
                st.error(f"💔 Damage applied to **`{layer}`** using **`{mode}`** mode.")

        # --- Detect Damaged Layer ---
        if st.button("🩻 Detect Damaged Layers"):
            if 'ref_outputs' not in st.session_state:
                st.warning("⚠️ Please save reference outputs first.")
            else:
                curr_op = get_layer_outputs(st.session_state.model, st.session_state.X_test[:10])
                diffs = compare_saved_outputs(st.session_state.ref_outputs, curr_op)
                layer_names = ['dense0', 'dense1', 'dense2', 'output']
                damaged = find_damaged_layer(diffs, layer_names)
                st.session_state.damaged_layer = damaged
                st.error(f"💥 Most Damaged Layer: **{damaged}**")

        # --- Accuracy After Damage ---
        if st.button("📉 Test Accuracy After Damage"):
            acc = get_acc(st.session_state.model, st.session_state.X_test, st.session_state.y_test_cat)
            st.metric("🎯 Test Accuracy", f"{acc:.2f} %")
            st.progress(acc / 100)

        # --- Heal Structurally Damaged Model ---
        if st.button("🩹 Heal the Model"):
            if 'damaged_layer' not in st.session_state:
                st.warning("⚠️ Damage the model first before healing.")
            else:
                with st.spinner("Healing in progress..."):
                    healed_model, _ = train_healing_patch(
                        st.session_state.model,
                        st.session_state.damaged_layer,
                        st.session_state.X_train,
                        st.session_state.y_train_cat
                    )
                    st.session_state.model = healed_model
                    st.success("✅ Healing complete! Model updated.")

        # --- Adversarial Training for Healing ---
        if st.button("🛡️ Heal Model with Adversarial Training"):
            try:
                with st.spinner("Generating adversarial samples and healing..."):
                    model = st.session_state.model
                    X_train = st.session_state.X_train
                    y_train_cat = st.session_state.y_train_cat

                    N = 9000
                    third = N // 3
                    idx_clean = np.random.choice(len(X_train), N, replace=False)
                    idx_fgsm = np.random.choice(len(X_train), third, replace=False)
                    idx_pgd = np.random.choice(len(X_train), third, replace=False)
                    idx_gen = np.random.choice(len(X_train), third, replace=False)

                    X_fgsm = fgsm_attack(model, X_train[idx_fgsm], y_train_cat[idx_fgsm])
                    X_pgd = pgd_attack(model, X_train[idx_pgd], y_train_cat[idx_pgd])
                    X_gen = generate_adversarial_examples(model, X_train[idx_gen], y_train_cat[idx_gen])

                    X_adv = np.concatenate([X_train[idx_clean], X_fgsm, X_pgd, X_gen], axis=0)
                    y_adv = np.concatenate([
                        y_train_cat[idx_clean],
                        y_train_cat[idx_fgsm],
                        y_train_cat[idx_pgd],
                        y_train_cat[idx_gen]
                    ], axis=0)

                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    model.fit(X_adv, y_adv, epochs=5, batch_size=128, validation_split=0.1, verbose=0)

                    st.session_state.model = model
                    st.success("✅ Model healed using adversarial examples!")
            except Exception as e:
                st.error(f"❌ Error during adversarial healing: {e}")

        # --- Post-Healing Adversarial Evaluation ---
        if st.button("📈 Re-evaluate After Adversarial Repair"):
            if not all(k in st.session_state for k in ['X_fgsm', 'X_pgd', 'X_gen', 'X_sample', 'y_sample']):
                st.warning("⚠️ Please evaluate on adversarial samples first.")
            else:
                model = st.session_state.model
                X_sample = st.session_state.X_sample
                y_sample = st.session_state.y_sample

                preds = {
                    "Normal": model.predict(X_sample),
                    "FGSM": model.predict(st.session_state.X_fgsm),
                    "PGD": model.predict(st.session_state.X_pgd),
                    "General": model.predict(st.session_state.X_gen)
                }

                accs = {k: np.mean(np.argmax(v, axis=1) == np.argmax(y_sample, axis=1)) for k, v in preds.items()}

                for k, v in accs.items():
                    st.metric(f"{k} Accuracy", f"{v * 100:.2f}%")

                st.subheader("📊 Post-Healing Accuracy Comparison")
                st.bar_chart({ "Accuracy (%)": {k: v * 100 for k, v in accs.items()} })

        # --- Final Test Accuracy ---
        if st.button("📊 Calculate Test Accuracy"):
            acc = get_acc(st.session_state.model, st.session_state.X_test, st.session_state.y_test_cat)
            st.metric("🎯 Test Accuracy", f"{acc:.2f} %")
            st.progress(acc / 100)

    except KeyError as e:
        st.error(f"❌ Missing expected key in .mat file: {e}")
    except Exception as e:
        st.error(f"⚠️ Error while processing file: {e}")
