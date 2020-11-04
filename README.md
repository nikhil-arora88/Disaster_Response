# Disaster Response Pipeline Project

### Objective:
The project aims to build a model that can predict the need for assistance by flagging the kind of help/aid required during a disaster from various messages (direct, social media etc.).


### Methodology:
The model uses NLP techniques and supervised machine learning. Each message is first cleaned and tokens created are created from the text message. From the clean tokens, features are created using the TF-IDF technique for feeding into a Random Forest model (trained using grid search)



### Instructions to run:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
