from keras.models import load_model
from preproc import preprocess_text
import pickle
import numpy as np
import requests
from time import localtime, strftime
import random

print('Please, wait till bot will be ready.')
model = load_model('chatbot_model.h5')
vectorizer = pickle.load(open("feature.pkl", "rb"))
varios = pickle.load(open('varios.pkl', 'rb'))
answers = pickle.load(open('answers.pkl', 'rb'))

answer = ''
print('Bot is ready to work!')
while answer != 'прощание':
    ask = input()
    ask = preprocess_text(ask)
    req = [ask]
    req = vectorizer.transform(req)
    req = [req]
    pred = model.predict(req)
    pred_label = np.argmax(pred)
    answer = varios[pred_label]
    if answer == 'погода':
        api_key = "0ad6cf474aac10f650993c303fb077f9"
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = 'Novosibirsk'
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        x = response.json()
        if x["cod"] != "404":
            y = x["main"]
            current_temperature = y["temp"] - 273.15
            current_pressure = y["pressure"]
            z = x["weather"]
            weather_description = z[0]["description"]
            print("Текущая температура = " +
                  str(current_temperature) +
                  "\n атмосферное давление = " +
                  str(current_pressure) +
                  "\n описание погоды: = " +
                  str(weather_description))

        else:
            print('Не получилось узнать погоду :(')
    elif answer == 'время':
        cur_time = strftime("%H:%M:%S", localtime())
        print(answers[pred_label][random.randint(0, len(answers[pred_label]) - 1)], cur_time)
    else:
        print(answers[pred_label][random.randint(0, len(answers[pred_label]) - 1)])
