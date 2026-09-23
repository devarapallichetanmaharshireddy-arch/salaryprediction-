
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("Salary Data.csv")
for c in ["Age","Years of Experience","Salary"]:
    df[c]=df[c].fillna(df[c].mean())
for c in ["Gender","Education Level","Job Title"]:
    df[c]=df[c].fillna(df[c].mode()[0])

le_gender=LabelEncoder()
le_edu=LabelEncoder()
le_job=LabelEncoder()

df["Gender"]=le_gender.fit_transform(df["Gender"])
df["Education Level"]=le_edu.fit_transform(df["Education Level"])
df["Job Title"]=le_job.fit_transform(df["Job Title"])

X=df.drop("Salary",axis=1)
y=df["Salary"]

model=RandomForestRegressor(n_estimators=200,max_depth=15,random_state=42)
model.fit(X,y)

joblib.dump(model,"salary_model.pkl")
joblib.dump((le_gender,le_edu,le_job,df),"encoders.pkl")
print("Model saved.")
