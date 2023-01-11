
import pickle
import train
import pandas as pd
import gradio as gr
import joblib
#clf = train.xgb_model
#clf.fit(train.df_train, train.y) 

#filename = "Completed_model.joblib"
#joblib.dump(clf, filename)

def make_prediction (date,parking,pool,children,city):
  loaded_model = joblib.load("model_xgb.joblib")
  #loaded_model = train.joblib.load(train.filename)
  if city in train.correspondance.to_string(index=False):
    city = float(train.correspondance[train.correspondance['city']==city]['encoding'].to_string(index=False))
    features = pd.DataFrame({
      'stock': [7],
      'date' : [date],
      'mobile' : [1],
      'parking' : [int(parking==True)],
      'pool' : [int(pool==True)],
      'children_policy' : [int(children==True)],
      'city_encoded' : [city],
      'language_encoded' : [159.865394],
      'group_encoded' : [227.055504],
      'brand_encoded' : [300.208264]
    })
    predictions = loaded_model.predict(features)
    #complete with what we want in the application
    return predictions[0]
  elif city == '':
    return "Nous n'avons pas d'hotel à vous proposer dans cette ville. Nous vous conseillons che prendre un hotel dans une des villes suivantes : amsterdam, vienna, madrid, rome, paris, valletta, vilnius, copenhagen, sofia"
  else:
    return "Nous n'avons pas d'hotel à vous proposer dans cette ville. Nous vous conseillons che prendre un hotel dans une des villes suivantes : amsterdam, vienna, madrid, rome, paris, valletta, vilnius, copenhagen, sofia"




#stock_input = gr.Number(label = "Stock")
date_input = gr.Number(label= "Nombre de jours avant votre voyage")
#mobile_input = gr.Number(label = "Mobile")
parking_input = gr.Checkbox(label = "Hotel avec parking?")
pool_input = gr.Checkbox(label = "Hotel avec piscine?")
children_input = gr.Checkbox(label = "Hotel avec club pour enfants?")
city_input = gr.Dropdown(["amsterdam","vienna","madrid","rome","paris","valletta","vilnus","copenhangen","sofia"],label = "Ville")
#language_input = gr.Number(label = "language")
#group_input = gr.Number(label = "group")
#brand_input = gr.Number(label = "brand")
# We create the output
output = gr.Text(label="Prix de l'hotel")


app = gr.Interface(fn = make_prediction, inputs=[date_input,parking_input,pool_input,
                                                 children_input,city_input], outputs=output,title="Prix de l'hotel par l'entreprise Seryana",
                                                 css="body {background-color: red}")
app.launch(share=True)
