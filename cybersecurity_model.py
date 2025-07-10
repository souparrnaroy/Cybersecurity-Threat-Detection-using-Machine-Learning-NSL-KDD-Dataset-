'''import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load NSL-KDD dataset (KDDTrain+ and KDDTest+)
train_data = pd.read_csv("KDDTrain+.txt", header=None)
test_data = pd.read_csv("KDDTest+.txt", header=None)

# Column names for NSL-KDD
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'  # ✅ Added this
]


train_data.columns = columns
test_data.columns = columns

# Convert attack labels into binary: normal vs attack
train_data['label'] = train_data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
test_data['label'] = test_data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# Combine datasets for encoding
combined_data = pd.concat([train_data, test_data])

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for col in categorical_cols:
    combined_data[col] = le.fit_transform(combined_data[col])

# Split back
train_data = combined_data[:len(train_data)]
test_data = combined_data[len(train_data):]

# Split features and target
X_train = train_data.drop(['label'], axis=1)
y_train = train_data['label']
X_test = test_data.drop(['label'], axis=1)
y_test = test_data['label']

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # <--- THIS LINE TRAINS THE MODEL

# Prediction
y_pred = model.predict(X_test)  # <--- Predict labels for test data

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
'''

# BETTER CODE USING STREAMLIT
'''import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Cybersecurity ML Analyzer", layout="wide")

st.title("🔐 Cybersecurity Attack Detection using Machine Learning")

# === Sidebar ===
st.sidebar.header("Upload Dataset")
uploaded_train = st.sidebar.file_uploader("Upload KDDTrain+.txt", type=["txt"])
uploaded_test = st.sidebar.file_uploader("Upload KDDTest+.txt", type=["txt"])

# Column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# === Load Data ===
@st.cache_data
def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)
    train_df.columns = columns
    test_df.columns = columns
    train_df = train_df.drop("difficulty_level", axis=1)
    test_df = test_df.drop("difficulty_level", axis=1)

    # Label attack vs normal
    train_df['label'] = train_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    test_df['label'] = test_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # Encode categorical
    combined = pd.concat([train_df, test_df])
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])
    
    return combined[:len(train_df)], combined[len(train_df):]

# Load data
if uploaded_train and uploaded_test:
    train_data, test_data = load_data(uploaded_train, uploaded_test)

    # Split features/labels
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]
    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # === Train model ===
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Model trained successfully!")

    # === Show Metrics ===
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🧩 Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['normal', 'attack'])
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=ax)
    st.pyplot(fig)

else:
    st.warning("📁 Please upload both KDDTrain+ and KDDTest+ files from NSL-KDD dataset.")


'''
#BETTER CUSTOMISE INPUT CODE
    # === CUSTOM INPUT PREDICTION ===
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Cybersecurity ML Analyzer", layout="wide")

st.title("🔐 Cybersecurity Attack Detection using Machine Learning")

# === Sidebar ===
st.sidebar.header("Upload Dataset")
uploaded_train = st.sidebar.file_uploader("Upload KDDTrain+.txt", type=["txt"])
uploaded_test = st.sidebar.file_uploader("Upload KDDTest+.txt", type=["txt"])

# === Column Names ===
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

# === Load Data ===
@st.cache_data
def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file, header=None)
    test_df = pd.read_csv(test_file, header=None)
    train_df.columns = columns
    test_df.columns = columns
    train_df = train_df.drop("difficulty_level", axis=1)
    test_df = test_df.drop("difficulty_level", axis=1)

    train_df['label'] = train_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    test_df['label'] = test_df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    combined = pd.concat([train_df, test_df])
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col])

    return combined[:len(train_df)], combined[len(train_df):]

# === Main App Logic ===
if uploaded_train and uploaded_test:
    train_data, test_data = load_data(uploaded_train, uploaded_test)

    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]
    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.success("✅ Model trained successfully!")

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("🧩 Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['normal', 'attack'])
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                ax=ax)
    st.pyplot(fig)

    # === Custom Input Prediction ===
    st.subheader("🧪 Test Custom Input")

    with st.form("custom_input_form"):
        st.markdown("Enter a sample input to test:")

        duration = st.number_input("Duration", min_value=0)
        protocol_type = st.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
        service = st.selectbox("Service", ['http', 'ftp', 'smtp', 'other'])
        flag = st.selectbox("Flag", ['SF', 'S0', 'REJ', 'SH'])

        src_bytes = st.number_input("Source Bytes", min_value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0)
        land = st.selectbox("Land", [0, 1])
        wrong_fragment = st.number_input("Wrong Fragment", min_value=0)
        urgent = st.number_input("Urgent", min_value=0)
        hot = st.number_input("Hot", min_value=0)
        num_failed_logins = st.number_input("Failed Logins", min_value=0)
        logged_in = st.selectbox("Logged In", [0, 1])

        remaining_features = [0] * (len(X_train.columns) - 12)

        # Get encoded values based on current training data
        pt_map = dict(zip(train_data['protocol_type'], train_data['protocol_type']))
        svc_map = dict(zip(train_data['service'], train_data['service']))
        flag_map = dict(zip(train_data['flag'], train_data['flag']))

        # Simple fallback: return 0 if not in training set
        pt_val = pt_map.get(protocol_type, 0)
        svc_val = svc_map.get(service, 0)
        flag_val = flag_map.get(flag, 0)

        input_vector = [
            duration, pt_val, svc_val, flag_val,
            src_bytes, dst_bytes, land, wrong_fragment,
            urgent, hot, num_failed_logins, logged_in
        ] + remaining_features

        predict_btn = st.form_submit_button("🚀 Predict")

        if predict_btn:
            input_array = np.array(input_vector).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            if prediction == "attack":
                st.error(f"🚨 ALERT: This input is predicted as an **ATTACK**.")
            else:
                st.success(f"✅ This input is predicted as **NORMAL**.")
else:
    st.warning("📁 Please upload both KDDTrain+ and KDDTest+ files.")
