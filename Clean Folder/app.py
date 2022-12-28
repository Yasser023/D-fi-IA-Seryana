
import pickle
import train
import pandas as pd
import gradio as gr

#clf = train.xgb_model
#clf.fit(train.df_train, train.y) 

#filename = "Completed_model.joblib"
#joblib.dump(clf, filename)

def make_prediction (stock,date,mobile,parking,pool,children,city,language,group,brand):
  loaded_model = train.joblib.load(train.filename)
  features = pd.DataFrame({
    'stock': [stock],
    'date' : [date],
    'mobile' : [mobile],
    'parking' : [parking],
    'pool' : [pool],
    'children_policy' : [children],
    'city_encoded' : [city],
    'language_encoded' : [language],
    'group_encoded' : [group],
    'brand_encoded' : [brand]
})
  predictions = loaded_model.predict(features)
  #complete with what we want in the application
  return predictions[0]

stock_input = gr.Number(label = "Stock")
date_input = gr.Number(label= "Date")
mobile_input = gr.Number(label = "Mobile")
parking_input = gr.Number(label = "Parking")
pool_input = gr.Number(label = "pool")
children_input = gr.Number(label = "children")
city_input = gr.Number(label = "city")
language_input = gr.Number(label = "language")
group_input = gr.Number(label = "group")
brand_input = gr.Number(label = "brand")
# We create the output
output = gr.Number()


app = gr.Interface(fn = make_prediction, inputs=[stock_input, date_input, mobile_input, parking_input,pool_input,
                                                 children_input,city_input,language_input,group_input,
                                                 brand_input], outputs=output)
app.launch(share=True)
