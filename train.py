import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import ConcreteModel
import joblib
import os

xls = pd.ExcelFile('Concrete_Data.xls')
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
df.columns = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Strength']

X = df.drop(columns=['Strength']).values
y = df['Strength'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем scaler на трейне
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Сохраняем scaler в папку model/
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

model = ConcreteModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    preds = model(torch.FloatTensor(X_train_scaled))
    loss = criterion(preds, torch.FloatTensor(y_train))
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), 'model/model.pth')
print("Модель и scaler успешно сохранены!")