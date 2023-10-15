import unittest
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
class TestMLCode(unittest.TestCase):
    def setUp(self):
        df = pd.read_csv("Iris.csv")
        df1=df.iloc[:,1:]

        x=df1.iloc[:,:-1]
        y=df1["Species"]

        le=LabelEncoder()
        y= le.fit_transform(y)
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(x,y,random_state=42,train_size=0.2)


    def test_model_predictions(self):
        trained_model = model.train_model(self.x_train,self.y_train)
        float_predictions = trained_model.predict(self.x_test)
    
        # Convert float predictions to integers
        int_predictions = [round(prediction) for prediction in float_predictions]
        print("test values:",self.y_test)
        print("predictede values:",int_predictions)
        self.assertListEqual(list(int_predictions), list(self.y_test))

if __name__ == '__main__':
    unittest.main()
