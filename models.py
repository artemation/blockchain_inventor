from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import relationship
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy.schema import Identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = Column(Integer, Identity(), primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    role = Column(String(50), nullable=False)  # Добавлено

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"

class Единица_измерения(db.Model):
    __tablename__ = 'Единица_измерения'

    Единица_ИзмеренияID = Column(Integer, Identity(), primary_key=True)
    Единица_Измерения = Column(String(255))

    def __repr__(self):
        return f"<Единица_измерения(Единица_ИзмеренияID={self.Единица_ИзмеренияID}, Единица_Измерения='{self.Единица_Измерения}')>"

class Склады(db.Model):
    __tablename__ = 'Склады'

    СкладID = Column(Integer, Identity(), primary_key=True)
    Название = Column(String(255))
    Адрес = Column(String(255))
    Телефон = Column(String(20))

    def __repr__(self):
        return f"<Склады(СкладID={self.СкладID}, Название='{self.Название}')>"

class Тип_документа(db.Model):
    __tablename__ = 'Тип_документа'

    ДокументID = Column(Integer, Identity(), primary_key=True)
    Тип_документа = Column(String(255))

    def __repr__(self):
        return f"<Тип_документа(ДокументID={self.ДокументID}, Тип_документа='{self.Тип_документа}')>"

class Товары(db.Model):
    __tablename__ = 'Товары'

    ТоварID = Column(Integer, Identity(), primary_key=True)
    Наименование = Column(String(255))
    Описание = Column(Text)
    Единица_ИзмеренияID = Column(ForeignKey('Единица_измерения.Единица_ИзмеренияID'))
    image_path = db.Column(db.String(255))

    Единица_измерения = relationship('Единица_измерения', foreign_keys=[Единица_ИзмеренияID])

    def __repr__(self):
        return f"<Товары(ТоварID={self.ТоварID}, Наименование='{self.Наименование}')>"

class Запасы(db.Model):
    __tablename__ = 'Запасы'

    ЗапасID = Column(Integer, Identity(), primary_key=True) #Убедитесь, что Identity работает корректно
    ТоварID = Column(ForeignKey('Товары.ТоварID'))
    СкладID = Column(ForeignKey('Склады.СкладID'))
    Количество = Column(Float)
    Дата_обновления = Column(Date)
    Единица_ИзмеренияID = Column(ForeignKey('Единица_измерения.Единица_ИзмеренияID'))

    Единица_измерения = relationship('Единица_измерения')
    Склады = relationship('Склады')
    Товары = relationship('Товары')

    def __repr__(self):
        return f"<Запасы(ЗапасID={self.ЗапасID}, ТоварID={self.ТоварID}, СкладID={self.СкладID}, Количество={self.Количество})>"

class ПриходРасход(db.Model):
    __tablename__ = 'ПриходРасход'

    ПриходРасходID = Column(Integer, Identity(), primary_key=True)
    СкладОтправительID = Column(ForeignKey('Склады.СкладID'))
    СкладПолучательID = Column(ForeignKey('Склады.СкладID'))
    ДокументID = Column(ForeignKey('Тип_документа.ДокументID'))
    ТоварID = Column(ForeignKey('Товары.ТоварID'))
    Количество = Column(Float)
    Единица_ИзмеренияID = Column(ForeignKey('Единица_измерения.Единица_ИзмеренияID'))
    TransactionHash = Column(String)
    Timestamp = db.Column(db.DateTime)
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

    id = Column(Integer, Identity(), primary_key=True)
    code = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    used_at = Column(DateTime, nullable=True)

    user = relationship('User', foreign_keys=[user_id])

    def __repr__(self):
        return f"<Invitation(code='{self.code}')>"


class BlockchainBlock(db.Model):
    __tablename__ = 'blockchain_blocks'

    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    transactions = db.Column(db.Text, nullable=False)  # JSON строка
    previous_hash = db.Column(db.String(64), nullable=False)
    hash = db.Column(db.String(64), nullable=False, unique=True)
    node_id = db.Column(db.Integer, nullable=False)
    confirmed = db.Column(db.Boolean, default=False)
    confirming_node_id = db.Column(db.Integer, nullable=False)  # ID узла, подтвердившего блок

    def __repr__(self):
        return f'<Block {self.index}>'
