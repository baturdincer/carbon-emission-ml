import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Model year, Engine size , cylinders, city, highway, combined l/100km, combined mpg, co2 rating, smog rating, CO2 emissions (g/km)(target)
traindata= pd.read_csv("train.csv")
del traindata["Model"]
del traindata["Make"]
del traindata["Transmission"]
traindata['Fuel type'] = np.where(traindata['Fuel type'] == "X", 1, traindata['Fuel type'])
traindata['Fuel type'] = np.where(traindata['Fuel type'] == "Z", 2, traindata['Fuel type'])
traindata['Fuel type'] = np.where(traindata['Fuel type'] == "E", 3, traindata['Fuel type'])
traindata['Fuel type'] = np.where(traindata['Fuel type'] == "D", 4, traindata['Fuel type'])
traindata['Fuel type'] = np.where(traindata['Fuel type'] == "N", 5, traindata['Fuel type'])
traindata["Fuel type"].astype("float64")
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Compact", 4, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Full-size", 6, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Sport utility vehicle: Small", 7, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Sport utility vehicle: Standard", 8, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Sport utility vehicle", 9, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Pickup truck: Standard", 12, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Mid-size", 5, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Subcompact", 3, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Minicompact", 2, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Station wagon: Small", 13, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Two-seater", 1, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Special purpose vehicle", 10, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Pickup truck: Small", 11, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Minivan", 15, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Station wagon: Mid-size", 14, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Van: Cargo", 17, traindata['Vehicle class'])
traindata['Vehicle class'] = np.where(traindata['Vehicle class'] == "Van: Passenger", 16, traindata['Vehicle class'])
traindata["Vehicle class"].astype("float64")
co2mean=traindata["CO2 rating"].mean()
smogratingmean=traindata["Smog rating"].mean()
traindata=traindata.fillna({"CO2 rating":co2mean})
traindata=traindata.fillna({"Smog rating": smogratingmean})

def computecost(w1, w2, w3, w4, w5, w6, w7, w8, w9,w10, b ,x1, x2, x3, x4, x5, x6, x7, x8, x9,x10, y):
    m=len(y)
    y_pred=w1*x1+w2*x2+w3*x3+w4*x4+w5*x5+w6*x6+w7*x7+w8*x8+w9*x9+w10*x10+b
    cost= (1/(2*m))*np.sum((y_pred-y)**2)
    return cost


def gradient_descent_path(x1,x2, x3, x4, x5, x6, x7, x8, x9,x10, y,w1_init, w2_init, w3_init, w4_init, w5_init, w6_init, w7_init, w8_init, w9_init, w10_init,b_init, learning_rate, iterations):
    m=len(y)
    w1,w2, w3, w4, w5, w6, w7, w8, w9, w10, b=w1_init, w2_init, w3_init, w4_init, w5_init, w6_init, w7_init, w8_init, w9_init, w10_init, b_init
    w1_history,w2_history, w3_history, w4_history, w5_history, w6_history, w7_history, w8_history, w9_history, w10_history, cost_history, b_history=[w1], [w2], [w3], [w4], [w5], [w6], [w7],[w8], [w9],[w10],[computecost(w1, w2, w3, w4, w5, w6, w7, w8, w9,w10, b ,x1, x2, x3, x4, x5, x6, x7, x8, x9,x10, y)],[b]
    for _ in range(iterations):
        y_pred= w1*x1+w2*x2+w3*x3+w4*x4+w5*x5+w6*x6+w7*x7+w8*x8+w9*x9+w10*x10+b

        dw1=  (1 / m) * np.sum((y_pred - y) * x1)
        dw2 = (1 / m) * np.sum((y_pred - y) * x2)
        dw3 = (1 / m) * np.sum((y_pred - y) * x3)
        dw4 = (1 / m) * np.sum((y_pred - y) * x4)
        dw5 = (1 / m) * np.sum((y_pred - y) * x5)
        dw6 = (1 / m) * np.sum((y_pred - y) * x6)
        dw7 = (1 / m) * np.sum((y_pred - y) * x7)
        dw8 = (1 / m) * np.sum((y_pred - y) * x8)
        dw9 = (1 / m) * np.sum((y_pred - y) * x9)
        dw10 = (1 / m) * np.sum((y_pred - y) * x10)
        db = (1 / m) * np.sum(y_pred - y)

        

        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        w3 -= learning_rate * dw3
        w4 -= learning_rate * dw4
        w5 -= learning_rate * dw5
        w6 -= learning_rate * dw6
        w7 -= learning_rate * dw7
        w8 -= learning_rate * dw8
        w9 -= learning_rate * dw9
        w10 -= learning_rate * dw10
        b -= learning_rate * db

        if(computecost(w1,w2, w3, w4, w5, w6, w7, w8, w9,w10, b,x1, x2, x3, x4, x5, x6, x7, x8, x9,x10, y)<cost_history[0]):
            w1_history[0]=w1
            w2_history[0]=w2
            w3_history[0]=w3
            w4_history[0]=w4
            w5_history[0]=w5
            w6_history[0]=w6
            w7_history[0]=w7
            w8_history[0]=w8
            w9_history[0]=w9
            w10_history[0]=w10
            b_history[0]=b
            cost_history[0]=(computecost(w1,w2, w3, w4, w5, w6, w7, w8, w9,w10, b,x1, x2, x3, x4, x5, x6, x7, x8, x9,x10, y))
    return  w1_history, w2_history, w3_history, w4_history, w5_history, w6_history, w7_history, w8_history, w9_history, w10_history, cost_history, b_history

x1=traindata["Fuel type"].to_numpy()
x2=traindata["Engine size (L)"].to_numpy()
x3=traindata["Cylinders"].to_numpy()
x4=traindata["City (L/100 km)"].to_numpy()
x5=traindata["Highway (L/100 km)"].to_numpy()
x6=traindata["Combined (L/100 km)"].to_numpy()
x7=traindata["Combined (mpg)"].to_numpy()
x8=traindata["CO2 rating"].to_numpy()
x9=traindata["Smog rating"].to_numpy()
x10=traindata["Vehicle class"].to_numpy()
y=traindata["CO2 emissions (g/km)"].to_numpy()

w1_init,w2_init, w3_init, w4_init, w5_init, w6_init, w7_init, w8_init, w9_init,w10_init, b_init=0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
learning_rate=0.001
iterations=100000
w1_history, w2_history, w3_history, w4_history, w5_history, w6_history, w7_history, w8_history, w9_history,w10_history, cost_history, b_history= gradient_descent_path(x1,x2, x3, x4, x5, x6, x7, x8, x9,x10, y,w1_init, w2_init, w3_init, w4_init, w5_init, w6_init, w7_init, w8_init, w9_init,w10_init, b_init, learning_rate, iterations)




idx=0
w1f=w1_history[idx]
w2f=w2_history[idx]
w3f=w3_history[idx]
w4f=w4_history[idx]
w5f=w5_history[idx]
w6f=w6_history[idx]
w7f=w7_history[idx]
w8f=w8_history[idx]
w9f=w9_history[idx]
w10f=w10_history[idx]
bf=b_history[idx]

td= pd.read_csv("test.csv")

del td["Model"]
del td["Make"]
del td["Transmission"]
co2mean=td["CO2 rating"].mean()
smogratingmean=td["Smog rating"].mean()
td=td.fillna({"CO2 rating":co2mean})
td=td.fillna({"Smog rating": smogratingmean})

td['Fuel type'] = np.where(td['Fuel type'] == "X", 1, td['Fuel type'])
td['Fuel type'] = np.where(td['Fuel type'] == "Z", 2, td['Fuel type'])
td['Fuel type'] = np.where(td['Fuel type'] == "E", 3, td['Fuel type'])
td['Fuel type'] = np.where(td['Fuel type'] == "D", 4, td['Fuel type'])
td['Fuel type'] = np.where(td['Fuel type'] == "N", 5, td['Fuel type'])
td["Fuel type"].astype("float64")


td['Vehicle class'] = np.where(td['Vehicle class'] == "Compact", 4, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Full-size", 6, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Sport utility vehicle: Small", 7, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Sport utility vehicle: Standard", 8, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Sport utility vehicle", 9, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Pickup truck: Standard", 12, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Mid-size", 5, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Subcompact", 3, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Minicompact", 2, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Station wagon: Small", 13, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Two-seater", 1, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Special purpose vehicle", 10, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Pickup truck: Small", 11, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Minivan", 15, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Station wagon: Mid-size", 14, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Van: Cargo", 17, td['Vehicle class'])
td['Vehicle class'] = np.where(td['Vehicle class'] == "Van: Passenger", 16, td['Vehicle class'])
td["Vehicle class"].astype("float64")


x1=td["Fuel type"].to_numpy()
x2=td["Engine size (L)"].to_numpy()
x3=td["Cylinders"].to_numpy()
x4=td["City (L/100 km)"].to_numpy()
x5=td["Highway (L/100 km)"].to_numpy()
x6=td["Combined (L/100 km)"].to_numpy()
x7=td["Combined (mpg)"].to_numpy()
x8=td["CO2 rating"].to_numpy()
x9=td["Smog rating"].to_numpy()
x10=td["Vehicle class"].to_numpy()

# w1f=-2.2842468182247693
# w2f=3.4695393957024363
# w3f=5.838711741047983
# w4f=6.684509706191565
# w5f=4.335736390818821
# w6f=5.631580682034835
# w7f=0.9160908407349152
# w8f=-1.3571998986177123
# w9f=0.904508811212
# bf=0.6868544091375172



df=pd.DataFrame(w1f*x1+w2f*x2+w3f*x3+w4f*x4+w5f*x5+w6f*x6+w7f*x7+w8f*x8+w9f*x9+w10f*x10+bf)
df.to_csv("deneme666.csv", index=True)
# print(w1f, w2f, w3f, w4f, w5f, w6f, w7f, w8f, w9f,bf)
print(cost_history[idx])