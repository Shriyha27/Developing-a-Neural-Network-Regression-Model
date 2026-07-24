# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: V. Shriyha

### Register Number: 212224230267

```python
import torch

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\admin\Downloads\Exp-1.csv")
df

x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)

scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)

xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)


class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1))
    def forward(self,x):
        return self.network(x)

model=neuralnet()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)

epochs = 1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred = model(xt)
    loss = criterion(pred, yt)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())

new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())

with torch.no_grad():
    pred=model(xst)
    loss_test=criterion(pred,yst)
    print(loss_test)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Over Time")
plt.show()



```

### Dataset Information
<img width="217" height="561" alt="image" src="https://github.com/user-attachments/assets/8846c26e-7f8b-44a0-ab14-05c388dde43f" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="712" height="568" alt="image" src="https://github.com/user-attachments/assets/d0d5b56e-c2a6-4dd5-b678-e9b15ab6d675" />


### New Sample Data Prediction
<img width="755" height="35" alt="image" src="https://github.com/user-attachments/assets/5f2fec4d-0775-4188-bc37-9acdf74a2546" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
