from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy.orm import relationship
from sqlalchemy.schema import Identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# app = Flask(__name__) # Убедитесь, что у вас есть экземпляр Flask-приложения
# app.config['SQLALCHEMY_DATABASE_URI'] = '...' # Укажите URI базы данных
db = SQLAlchemy() # app = app  Удалите app=app если вы инициализируете db вне функции create_app

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, Identity(), primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(50), nullable=False)  # Добавлено

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

class Единица_измерения(db.Model):
    __tablename__ = 'Единица_измерения'

    Единица_ИзмеренияID = db.Column(db.Integer, Identity(), primary_key=True)
    Единица_Измерения = db.Column(db.String(255))

    def __repr__(self):
        return f"<Единица_измерения(Единица_ИзмеренияID={self.Единица_ИзмеренияID}, Единица_Измерения='{self.Единица_Измерения}')>"

class Склады(db.Model):
    __tablename__ = 'Склады'

    СкладID = db.Column(db.Integer, Identity(), primary_key=True)
    Название = db.Column(db.String(255))
    Адрес = db.Column(db.String(255))
    Телефон = db.Column(db.String(20))

    def __repr__(self):
        return f"<Склады(СкладID={self.СкладID}, Название='{self.Название}')>"

class Тип_документа(db.Model):
    __tablename__ = 'Тип_документа'

    ДокументID = db.Column(db.Integer, Identity(), primary_key=True)
    Тип_документа = db.Column(db.String(255))

    def __repr__(self):
        return f"<Тип_документа(ДокументID={self.ДокументID}, Тип_документа='{self.Тип_документа}')>"

class Товары(db.Model):
    __tablename__ = 'Товары'

    ТоварID = db.Column(db.Integer, Identity(), primary_key=True)
    Наименование = db.Column(db.String(255))
    Описание = db.Column(db.Text)
    Единица_ИзмеренияID = db.Column(db.ForeignKey('Единица_измерения.Единица_ИзмеренияID'))
    image_path = db.Column(db.String(255))

    Единица_измерения = relationship('Единица_измерения', foreign_keys=[Единица_ИзмеренияID])

    def __repr__(self):
        return f"<Товары(ТоварID={self.ТоварID}, Наименование='{self.Наименование}')>"

class Запасы(db.Model):
    __tablename__ = 'Запасы'

    ЗапасID = db.Column(db.Integer, Identity(), primary_key=True) #Убедитесь, что Identity работает корректно
    ТоварID = db.Column(db.ForeignKey('Товары.ТоварID'))
    СкладID = db.Column(db.ForeignKey('Склады.СкладID'))
    Количество = db.Column(db.Float)
    Дата_обновления = db.Column(db.Date)
    Единица_ИзмеренияID = db.Column(db.ForeignKey('Единица_измерения.Единица_ИзмеренияID'))

    Единица_измерения = relationship('Единица_измерения')
    Склады = relationship('Склады')
    Товары = relationship('Товары')

    def __repr__(self):
        return f"<Запасы(ЗапасID={self.ЗапасID}, ТоварID={self.ТоварID}, СкладID={self.СкладID}, Количество={self.Количество})>"

class ПриходРасход(db.Model):
    __tablename__ = 'ПриходРасход'

    ПриходРасходID = db.Column(db.Integer, Identity(), primary_key=True)
    СкладОтправительID = db.Column(db.ForeignKey('Склады.СкладID'))
    СкладПолучательID = db.Column(db.ForeignKey('Склады.СкладID'))
    ДокументID = db.Column(db.ForeignKey('Тип_документа.ДокументID'))
    ТоварID = db.Column(db.ForeignKey('Товары.ТоварID'))
    Количество = db.Column(db.Float)
    Единица_ИзмеренияID = db.Column(db.ForeignKey('Единица_измерения.Единица_ИзмеренияID'))
    TransactionHash = db.Column(db.String(64), unique=True, nullable=False)  # <- Вот эта строка изменена
    Timestamp = db.Column(db.DateTime(timezone=False))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    user = db.relationship('User', backref='transactions')
    Тип_документа = relationship('Тип_документа')
    СкладОтправитель = relationship('Склады', foreign_keys=[СкладОтправительID])
    СкладПолучатель = relationship('Склады', foreign_keys=[СкладПолучательID])
    Товары = relationship('Товары')
    Единица_измерения = relationship('Единица_измерения')

    def __repr__(self):
        return f"<ПриходРасход(ПриходРасходID={self.ПриходРасходID}, СкладОтправительID={self.СкладОтправительID}, СкладПолучательID={self.СкладПолучательID}, ТоварID={self.ТоварID}, Количество={self.Количество})>"
        
class Invitation(db.Model):
    __tablename__ = 'invitations'

    id = db.Column(db.Integer, Identity(), primary_key=True)
    code = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    used_at = db.Column(db.DateTime, nullable=True)

    user = relationship('User', foreign_keys=[user_id])

    def __repr__(self):
        return f"<Invitation(code='{self.code}')>"


class BlockchainBlock(db.Model):
    __tablename__ = 'blockchain_blocks'

    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False)
    transactions = db.Column(db.Text, nullable=False)  # JSON строка
    previous_hash = db.Column(db.String(64), nullable=False)
    hash = db.Column(db.String(64), nullable=False, unique=True)
    node_id = db.Column(db.Integer, nullable=False)
    confirmed = db.Column(db.Boolean, default=False)
    confirming_node_id = db.Column(db.Integer, nullable=False)  # ID узла, подтвердившего блок
    confirmations = db.Column(db.Text)

    def __repr__(self):
        return f'<Block {self.index}>'

    def __repr__(self):
        return f'<Block {self.index}>'
