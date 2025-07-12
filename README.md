# Pitch Analyzer 
An intelligent machine learning tool to analyze startup pitch delivery - 'Problem/Solution', 'Market', 'Traction', 'Business model', 'Team', 'Competition', 'Projections'
##Features -
**Voice Recognition** using Google's Speech Recognition API  
- **Text Cleaning** with NLTK (punctuation removal, stopword filtering)  
- **TF-IDF Vectorization** to represent pitch content numerically  
- **ML Models**:
  - **Category Classifier** (Random Forest)
  - **Strategy Classifier** (Random Forest)
  - **7 Stat Score Predictors** (Problem/Solution, Market, Traction, Business Model, Team, Competition, Projections)
- **Multi-Mode Input**:
  - **Typed pitch**
  - **Microphone input (live speech)**
  - **Audio file (.wav)**
**Project Structure**-
```bash
PitchAnalyzer/
├── main.py
├── requirements.txt
├── README.md
├── model/
│   ├── trained_models.pkl
├── data/
│   └── Startup_Idea_Analyzer_FILLED_FINAL.csv
```
## Setup Instructions

**Install Dependencies**
  ```bash
   pip install -r pitchrequirements.txt
   If PyAudio fails, install it using pipwin:
   pip install pipwin
   pipwin install pyaudio
  ```
**Required Python Libraries**
 - **speechrecognition**
 - **nltk**
 - **pandas**
 - **scikit-learn**
 - **pyaudio**

**Download NLTK Data**
import nltk
```bash
nltk.download('punkt')
nltk.download('stopwords')
```
**How to use**-
Run the script:
```bash
python pitch_analyzer.py
```
**Select input method**:
 - **typed: Manually type your pitch**
 - **mic: Record pitch via microphone**
 - **file: Use an existing .wav file**

**Output**:
- Predicted Category
- Business Strategy
- 7 Scored Metrics (/10)
- Final Viability Score

**Example:**
 
 Predictions for your pitch:
 
 Category: EdTech
 
 Strategy: Personalized Learning
 
 Problem/Solution: 8.7/10
 
 Market: 9.2/10
 
 Traction: 7.6/10
 
 Business model: 8.4/10
 
 Team: 9.0/10
 
 Competition: 6.8/10
 
 Projections: 8.1/10

 Viability Score is 8.11

## Model Training Notes-
- All models are trained on TF-IDF features.
- Stat predictors are individual regressors.
- Category & Strategy are label-encoded classifiers.
- Dataset is cleaned and normalized before training.

**Author**

Vinit Malgunde

