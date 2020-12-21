import pickle
import numpy as np


"""
Load the model and scalers
"""
def load_model():
    x_scaler = pickle.load(open("X_scaler.bin", 'rb'))
    y_scaler = pickle.load(open("Y_scaler.bin", 'rb'))
    loaded_model = pickle.load(open("MODEL.bin", 'rb'))

    return x_scaler, y_scaler, loaded_model

"""
get input from USER
"""
def get_input():
    pr = float(input("Please enetr the value for PR:  "))
    ar = float(input("Please enetr the value for AR:  "))
    height = int(input("Please enetr the value for Height:  "))

    print("\n You successfully enetred the values \n")

    return pr, ar, height

"""
Main Function 
"""
def main():

    x_scaler, y_scaler, loaded_model = load_model()
    pr, ar, height = get_input()
    #transform input
    X = x_scaler.transform(np.array([pr, ar, height]).reshape(1, -1))

    #predict
    predict_y = loaded_model.predict(X).reshape(-1,1)
    print("=============== Estimating ... ================== \n")

    #inverse transform y
    predict_y = y_scaler.inverse_transform(predict_y)

    print("Estimation of Velocity = ", round(predict_y.ravel()[0], 6))


if __name__ == "__main__":
    main()
