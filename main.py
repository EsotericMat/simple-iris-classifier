from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)
model_pickled = pickle.load(open('iris_c_v1.pkl', 'rb'))

def predict(model, values):
  irises = {
      0:'Setosa',
      1:'Versicolor',
      2:'Virginica'
  }
  idx = model.predict(np.array([values]))[0]
  return irises.get(idx)


@app.route('/predict', methods=["GET","POST"])
def predict_the_iris():
  if request.method=="GET":
    return 'send request'

  elif request.method == "POST":
    data = request.get_json()
    sp_lng = data['a']
    sp_wid = data['b']
    pt_lng = data['c']
    pt_wid = data['d']
    in_ = np.array([sp_lng, sp_wid, pt_lng, pt_wid])
    return str(predict(model_pickled, in_))

def main():
  app.run()


if __name__ == '__main__':
    main()