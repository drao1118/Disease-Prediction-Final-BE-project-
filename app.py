import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

classifier = pickle.load(open('diabetes-prediction-rfc-model.pkl', 'rb'))
livermodel = pickle.load(open('livermodel.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))
        return render_template("login.html", form=form)
    return render_template("login.html", form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")
    return render_template('signup.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")

@app.route("/liver")
@login_required
def liver():
    return render_template("liver.html")

@app.route("/disblog")
@login_required
def disblog():
    return render_template("disblog.html")

@app.route("/diabetesMorF")
@login_required
def diabetesMorF():
    
    return render_template("diabetesMorF.html")

@app.route("/diabetes", methods=['POST'])
@login_required
def diabetes():
    gender = [x for x in request.form.values()]
    print(gender)
    # form = MyForm(request.POST)
    # print(form.data['gender'])
    if gender[0]== 'M':
        return render_template("diabetesM.html")
    else: 
        return render_template("diabetesF.html")



###########################################################################

@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age",  "sex",  "cp",  "trestbps",  "chol",  "fbs",  "restecg",  "thalach",  
                    "exang",  "oldpeak","slope" ,"ca", "hal"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    print("Prediction: ")
    print(output)
    print(type(output))
    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))

##################################################################################################################

@app.route('/predictliver', methods=['POST'])
def predictliver():
    input_features = [float(x) for x in request.form.values()]
    print("mayank:")
    print(input_features)
    if input_features[9]==1:
        input_features[9]=0
        input_features.append(1)
    else:
        input_features[9]=1
        input_features.append(0)
#female=0, male=1
    features_value = [np.array(input_features)]

    features_name = ["Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                    "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio","Gender_Female","Gender_Male"]

    df = pd.DataFrame(features_value, columns=features_name)
    print("tolele : ")
    print(df)
    result = livermodel.predict(df)
    print("Prediction: ")
    print(result)
    print(type(result))
    if result == 1:
        res_val = "a high risk of Liver Disease"
    else:
        res_val = "a low risk of Liver Disease"

    return render_template('liver_result.html', prediction_text='Patient has {}'.format(res_val))

##################################################################################################################

@app.route('/predictdiabetes', methods=['POST'])
def predictdiabetes():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["Pregnancies",	"Glucose",	"BloodPressure",	"SkinThickness",
                     "Insulin",	"BMI",	"DiabetesPedigreeFunction",	"Age"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = classifier.predict(df)
    print("Prediction: ")
    print(output)
    print(type(output))
    if output == 1:
        res_val = "a high risk of Diabetes"
    else:
        res_val = "a low risk of Diabetes"

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))

##################################################################################################################

if __name__ == "__main__":
    app.run(debug=True)

