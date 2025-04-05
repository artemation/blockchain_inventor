from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField, FloatField, SelectField, EmailField
from wtforms.validators import DataRequired, Length, EqualTo, ValidationError
from models import User, Склады, Тип_документа, Товары, Единица_измерения, Invitation

class RegistrationForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=4, max=255)])
    password = PasswordField('Пароль', validators=[DataRequired(), Length(min=6, max=255)])
    confirm_password = PasswordField('Подтвердите пароль', validators=[DataRequired(), EqualTo('password')])
    role = SelectField('Роль', choices=[('admin', 'Администратор'), ('north', 'Склад на севере'), ('south', 'Склад на юге')], validators=[DataRequired()])
    invitation_code = StringField('Код приглашения', validators=[DataRequired()])
    submit = SubmitField('Зарегистрироваться')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Это имя пользователя уже занято. Пожалуйста, выберите другое.')

    def validate_invitation_code(self, invitation_code):
        invitation = Invitation.query.filter_by(code=invitation_code.data, user_id=None).first()
        if not invitation:
            raise ValidationError('Неверный код приглашения.')

class LoginForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    submit = SubmitField('Войти')

class PrihodRashodForm(FlaskForm):
    СкладОтправительID = SelectField('Склад отправитель', coerce=int, validators=[DataRequired()])
    СкладПолучательID = SelectField('Склад получатель', coerce=int, validators=[DataRequired()])
    ДокументID = SelectField('Тип документа', coerce=int, validators=[DataRequired()])
    ТоварID = SelectField('Товар', coerce=int, validators=[DataRequired()])
    Количество = FloatField('Количество', validators=[DataRequired()])
    Единица_ИзмеренияID = SelectField('Единица измерения', coerce=int, validators=[DataRequired()])
    submit = SubmitField('Добавить')

    def __init__(self, *args, **kwargs):
        super(PrihodRashodForm, self).__init__(*args, **kwargs)
        self.СкладОтправительID.choices = [(s.СкладID, s.Название) for s in Склады.query.all()]
        self.СкладПолучательID.choices = [(s.СкладID, s.Название) for s in Склады.query.all()]
        self.ДокументID.choices = [(d.ДокументID, d.Тип_документа) for d in Тип_документа.query.all()]
        self.ТоварID.choices = [(t.ТоварID, t.Наименование) for t in Товары.query.all()]
        self.Единица_ИзмеренияID.choices = [(e.Единица_ИзмеренияID, e.Единица_Измерения) for e in Единица_измерения.query.all()]

class InvitationForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired()])
    submit = SubmitField('Создать приглашение')