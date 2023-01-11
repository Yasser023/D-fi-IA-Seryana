
import pickle
import train
import pandas as pd
import gradio as gr
import joblib
#clf = train.xgb_model
#clf.fit(train.df_train, train.y) 

#filename = "Completed_model.joblib"
#joblib.dump(clf, filename)

def make_prediction (stock,date,parking,pool,children,city,language,group,brand):
  loaded_model = joblib.load("model_xgb.joblib")
  #loaded_model = train.joblib.load(train.filename)
  if city in train.correspondance.to_string(index=False):
    city = float(train.correspondance[train.correspondance['city']==city]['encoding'].to_string(index=False))
    language = float(train.correspondance_language[train.correspondance_language['language']==language]['encoding'].to_string(index=False))
    group = float(train.correspondance_group[train.correspondance_group['group']==group]['encoding'].to_string(index=False))
    brand = float(train.correspondance_brand[train.correspondance_brand['brand']==brand]['encoding'].to_string(index=False))
    features = pd.DataFrame({
      'stock': [stock],
      'date' : [date],
      'mobile' : [1],
      'parking' : [int(parking==True)],
      'pool' : [int(pool==True)],
      'children_policy' : [int(children==True)],
      'city_encoded' : [city],
      'language_encoded' : [language],
      'group_encoded' : [group],
      'brand_encoded' : [brand]
    })
    predictions = loaded_model.predict(features)
    #complete with what we want in the application
    return predictions[0]
  elif city == '':
    return "Nous n'avons pas d'hotel à vous proposer dans cette ville. Nous vous conseillons de prendre un hotel dans une des villes suivantes : amsterdam, vienna, madrid, rome, paris, valletta, vilnius, copenhagen, sofia"
  else:
    return "Nous n'avons pas d'hotel à vous proposer dans cette ville. Nous vous conseillons de prendre un hotel dans une des villes suivantes : amsterdam, vienna, madrid, rome, paris, valletta, vilnius, copenhagen, sofia"




stock_input = gr.Number(label = "Stock")
date_input = gr.Number(label= "Nombre de jours avant votre voyage")
#mobile_input = gr.Number(label = "Mobile")
parking_input = gr.Checkbox(label = "Hotel avec parking?")
pool_input = gr.Checkbox(label = "Hotel avec piscine?")
children_input = gr.Checkbox(label = "Hotel avec club pour enfants?")
city_input = gr.Dropdown(["amsterdam","vienna","madrid","rome","paris","valletta","vilnius","copenhangen","sofia"],label = "Ville")
language_input = gr.Dropdown(['austrian', 'belgian', 'bulgarian', 'croatian', 'cypriot',
 'czech', 'danish', 'dutch', 'estonian', 'finnish', 'french', 'german', 'greek', 'hungarian', 'irish', 'italian', 'latvian',
  'lithuanian', 'luxembourgish', 'maltese', 'polish', 'portuguese', 'romanian', 'slovakian', 'slovene', 'spanish', 'swedish'],label = "Langue")
group_input = gr.Dropdown(["Independant","Accar Hotels","Chillton Worldwide","Morriott International","Boss Western", "Yin Yang"],label = "group")
brand_input = gr.Dropdown(["Independant","Marcure","Safitel","Tripletree","Chill Garden Inn","Corlton","J.Halliday Inn",
"Navatel","Boss Western","Quadrupletree","8 Premium","Ibas","Ardisson","CourtYord","Royal Lotus","Morriot"],label = "brand")
# We create the output
output = gr.Text(label="Prix de l'hotel")


app = gr.Interface(fn = make_prediction, inputs=[stock_input,date_input,parking_input,pool_input,
                                                 children_input,city_input,language_input,group_input,brand_input], outputs=output,title="Booking by Seryana",
                                                 css="body {background-color: red}")
app.launch(share=True)
