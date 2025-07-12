
import speech_recognition as sr
import string
import nltk
import pandas as pd
import threading
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# If punkt is not getting downloaded download it manually using --> import nltk , nltk.download()
nltk.download('punkt'),
nltk.download('stopwords')

# There may be diifferent situations to deal with so we have to be flexible with all kinds of input i.e. a typed pitch, an audio file or a mic input for live session
def get_pitch_input(mode="typed", audio_path=None):
    recognizer = sr.Recognizer()
    text = ""

    if mode == "typed":
        text = input("✍️ Enter your startup pitch: ")

    elif mode == "mic":
        print("🎤 Speak your startup pitch... Press ENTER to stop recording")
        stop_recording = False
        audio_data = []

        def listen_in_background():
            nonlocal audio_data
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_data.append(recognizer.listen(source, phrase_time_limit=None))

        def wait_for_enter():
            nonlocal stop_recording
            input()
            stop_recording = True

        threading.Thread(target=wait_for_enter).start()
        listen_in_background()
        while not stop_recording:
            pass

        try:
            text = recognizer.recognize_google(audio_data[0])
            print("📝 Transcribed Pitch:\n", text)
        except:
            text = ""

    elif mode == "file":
        if not audio_path:
            audio_path = input("📁 Path to .wav file: ")
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except:
            text = ""
    else:
        raise ValueError("Invalid input mode.")

    return text

# Cleaner
def clean_pitch(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stopwords.words('english')])

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\HTML CODING\Startup_Idea_Analyzer_FILLED_FINAL.csv")
df = df.drop(columns=['avg', 'multiplication factor', 'P', 'M', 'T', 'B', 'A', 'C', 'O'], errors='ignore')
df['Pitch'] = df['Pitch'].fillna("")
df['clean_pitch'] = df['Pitch'].apply(clean_pitch)

# Vectorize
tfidf = TfidfVectorizer(max_features=300)
X_all = tfidf.fit_transform(df['clean_pitch'])
X_df = pd.DataFrame(X_all.toarray(), columns=tfidf.get_feature_names_out())

# Encode Category,Strategy
le_cat = LabelEncoder()
le_strat = LabelEncoder()
df['Category'] = df['Category'].fillna("Unknown")
df['Strategy'] = df['Strategy'].fillna("Unknown")
y_cat = le_cat.fit_transform(df['Category'])
y_strat = le_strat.fit_transform(df['Strategy'])

# Stat score columns
stat_cols = ['Problem/Solution', 'Market', 'Traction', 'Business model', 'Team', 'Competition', 'Projections']
df[stat_cols] = df[stat_cols].fillna(df[stat_cols].mean())

# Train models
X_train, X_test, y_cat_train, y_cat_test = train_test_split(X_df, y_cat, test_size=0.2, random_state=42)
cat_model = RandomForestClassifier(n_estimators=150).fit(X_train, y_cat_train)

X_train, X_test, y_strat_train, y_strat_test = train_test_split(X_df, y_strat, test_size=0.2, random_state=42)
strat_model = RandomForestClassifier(n_estimators=150).fit(X_train, y_strat_train)

# Regression models
reg_models = {}
for col in stat_cols:
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150).fit(X_train, y_train)
    reg_models[col] = model

# Prediction
mode = input("Select input mode (typed/mic/file): ").lower().strip()
pitch = get_pitch_input(mode)
cleaned = clean_pitch(pitch)
vec = tfidf.transform([cleaned]).toarray()

cat_pred = le_cat.inverse_transform(cat_model.predict(vec))[0]
strat_pred = le_strat.inverse_transform(strat_model.predict(vec))[0]
stat_preds = {col: round(reg_models[col].predict(vec)[0], 2) for col in stat_cols}

# Output
print("\n🔮 Predictions for your pitch:")
print(f"📁 Category: {cat_pred}")
print(f"🧠 Strategy: {strat_pred}")
sumval=0
for col, val in stat_preds.items():
    print(f"📊 {col}: {val}/10")
    sumval = sumval+val
print("  Viability Score is",sumval/7)
