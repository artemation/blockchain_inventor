import os
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, current_app
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from forms import RegistrationForm, LoginForm, PrihodRashodForm, InvitationForm
import hashlib
from dotenv import load_dotenv
from models import db, Единица_измерения, Склады, Тип_документа, Товары, Запасы, ПриходРасход, User, Invitation, BlockchainBlock
import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter, StreamHandler, DEBUG
import json
import sqlalchemy
from sqlalchemy import text, select, func
from flask_wtf.csrf import CSRFProtect
import threading
from datetime import datetime, timezone
import psycopg2
import asyncio
import aiohttp
from uuid import uuid4
from flask_cors import CORS
from flask_session import Session
import time
from functools import wraps
from threading import Lock
import atexit
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()
# Определяем NODE_ID в начале
NODE_ID = int(os.environ.get('NODE_ID', 0))

NODE_DOMAINS = {
    0: os.environ.get('NODE0_DOMAIN', 'blockchaininventory0.up.railway.app'),
    1: os.environ.get('NODE1_DOMAIN', 'blockchaininventory1.up.railway.app'),
    2: os.environ.get('NODE2_DOMAIN', 'blockchaininventory2.up.railway.app'),
    3: os.environ.get('NODE3_DOMAIN', 'blockchaininventory3.up.railway.app')
}

app = Flask(__name__)
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%d.%m.%Y %H:%M'):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            return value
    return value.strftime(format)
csrf = CSRFProtect(app)
CORS(app)

app.logger.setLevel(logging.DEBUG)  # Убедитесь, что уровень логирования установлен на DEBUG

file_handler = RotatingFileHandler('app.log', maxBytes=1024 * 1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)  # Убедитесь, что уровень логирования для обработчика также установлен на DEBUG
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)

app.logger.debug('This is a test log message at the beginning')

db_user = os.environ.get('DB_USER')
db_password = os.environ.get('DB_PASSWORD')
db_host = os.environ.get('DB_HOST')
db_port = os.environ.get('DB_PORT', '5432')
db_name = os.environ.get('DB_NAME')

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://', 1)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_default_secret_key')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WTF_CSRF_ENABLED'] = True
app.config['NODE_ID'] = NODE_ID

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Функция для отключения CSRF для определенных маршрутов


@login_manager.user_loader
def load_user(user_id):
    return db.session.execute(select(User).filter_by(id=user_id)).scalar_one_or_none()

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'}), 200


@app.route('/get_blockchain_height')

def get_blockchain_height():
    """Вернуть высоту (индекс последнего блока) блокчейна"""
    try:
        last_block = BlockchainBlock.query.order_by(BlockchainBlock.index.desc()).first()
        height = last_block.index if last_block else -1
        return jsonify({'height': height}), 200
    except Exception as e:
        app.logger.error(f"Error in get_blockchain_height: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_block/<int:block_index>')

def get_block(block_index):
    """Вернуть блок с указанным индексом"""
    try:
        block = BlockchainBlock.query.filter_by(index=block_index).first()
        if not block:
            return jsonify({'error': 'Block not found'}), 404

        # Преобразуем блок в словарь
        transactions = json.loads(block.transactions)
        block_data = {
            'index': block.index,
            'timestamp': block.timestamp.isoformat(),
            'transactions': transactions,
            'previous_hash': block.previous_hash,
            'hash': block.hash,
            'node_id': block.node_id
        }
        return jsonify(block_data), 200
    except Exception as e:
        app.logger.error(f"Error in get_block: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin', methods=['GET'])
@login_required
def admin_panel():
    app.logger.debug('Entering admin_panel function')
    if not current_user.is_admin:
        flash('У вас нет прав для доступа к этой странице', 'danger')
        return redirect(url_for('index'))

    history_stats = {}
    for node_id, node in nodes.items():
        history_stats[node_id] = len(node.chain) - 1

    app.logger.debug('Exiting admin_panel function')
    return render_template('admin.html', history_stats=history_stats)

@app.route('/admin/clear_history', methods=['POST'])
@login_required
def clear_transaction_history():
    app.logger.debug('Entering clear_transaction_history function')
    if not current_user.is_admin:
        return jsonify({'success': False, 'message': 'Недостаточно прав'})

    node_id = request.form.get('node_id')
    older_than = request.form.get('older_than')

    if node_id == 'all':
        for node in nodes.values():
            if older_than:
                cutoff_date = datetime.now(timezone.utc) - datetime.timedelta(days=int(older_than))
                keys_to_delete = []

                for block in node.chain:
                    if block.timestamp < cutoff_date:
                        keys_to_delete.append(block.index)

                node.chain = [block for block in node.chain if block.index not in keys_to_delete]
            else:
                node.chain = [node.create_genesis_block()]
    else:
        node = nodes.get(int(node_id))
        if node:
            if older_than:
                cutoff_date = datetime.now(timezone.utc) - datetime.timedelta(days=int(older_than))
                keys_to_delete = []

                for block in node.chain:
                    if block.timestamp < cutoff_date:
                        keys_to_delete.append(block.index)

                node.chain = [block for block in node.chain if block.index not in keys_to_delete]
            else:
                node.chain = [node.create_genesis_block()]

    app.logger.debug('Exiting clear_transaction_history function')
    return jsonify({'success': True, 'message': 'История транзакций очищена'})


class Block:
    def __init__(self, index, timestamp, transactions, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    @staticmethod
    def calculate_hash(block_data):
        data = {
            'index': block_data['index'],
            'timestamp': block_data['timestamp'],
            'transactions': block_data['transactions'],
            'previous_hash': block_data['previous_hash']
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()

    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp.isoformat(),
            'transactions': self.transactions,
            'previous_hash': self.previous_hash,
            'hash': self.hash
        }

    @staticmethod
    def from_dict(block_dict):
        block = Block(
            index=block_dict['index'],
            timestamp=datetime.datetime.fromisoformat(block_dict['timestamp']),
            transactions=block_dict['transactions'],
            previous_hash=block_dict['previous_hash']
        )
        # Восстановить хеш из словаря, если он отличается от рассчитанного
        if block_dict.get('hash') and block_dict['hash'] != block.hash:
            block.hash = block_dict['hash']
        return block

# Настройка логгера для класса Node
node_logger = logging.getLogger('node')
node_logger.setLevel(logging.DEBUG)
node_handler = logging.handlers.RotatingFileHandler('node.log', maxBytes=1024*1024, backupCount=5)
node_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
node_logger.addHandler(node_handler)

class Node:
    # Константы для таймеров
    LEADER_TIMEOUT = 30  # Время ожидания ответа от лидера (в секундах)
    VIEW_CHANGE_TIMEOUT = 5  # Таймаут для процесса смены вида

    def __init__(self, node_id, nodes, host, port):
        self.node_id = node_id
        self.nodes = nodes
        self.host = host
        self.port = port
        self.sequence_number = 0
        self.prepared = {}
        self.committed = {}
        self.log = []
        self.view_number = 0
        self.is_leader = (node_id == 0)
        self.requests = {}
        self.chain = []
        self.leader_timeout = None
        self.view_change_in_progress = False
        self.consensus_times = []  # Список времен достижения консенсуса
        self.view_change_success = []  # Список результатов смены вида (True/False)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.sync_genesis_block())
        self.loop.run_until_complete(self.sync_view_number())
        self.start_leader_timeout()

    def start_leader_timeout(self):
            async def periodic_check():
                while True:
                    await self.check_leader_activity()
                    await asyncio.sleep(5)  # Проверять каждые 5 секунд
            self.loop.create_task(periodic_check())

    async def check_leader_activity(self):
        """
        Периодически проверяет активность лидера через запрос к /health.
        Если лидер недоступен, инициирует процесс смены лидера.
        """
        try:
            # Определяем ID лидера
            leader_id = self.view_number % (len(self.nodes) + 1)
            if leader_id == self.node_id:
                node_logger.debug(f"Узел {self.node_id} является лидером, проверка активности не требуется")
                return
    
            node_logger.debug(f"Узел {self.node_id} проверяет активность лидера {leader_id}")
            url = f"https://{self.nodes[leader_id]}/health"
    
            # Выполняем HTTP-запрос к эндпоинту /health лидера
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        node_logger.debug(f"Лидер {leader_id} активен, статус ответа: {response.status}")
                    else:
                        node_logger.warning(f"Лидер {leader_id} не отвечает, статус ответа: {response.status}")
                        node_logger.info(f"Узел {self.node_id} инициирует смену лидера")
                        await self.initiate_view_change()
    
        except Exception as e:
            node_logger.error(f"Узел {self.node_id} не смог проверить активность лидера {leader_id}: {str(e)}")
            node_logger.info(f"Узел {self.node_id} инициирует смену лидера из-за ошибки")
            await self.initiate_view_change()

    async def initiate_view_change(self):
        """Инициирует смену вида, рассылая запросы другим узлам"""
        if self.view_change_in_progress:
            node_logger.debug(f"Node {self.node_id}: View change already in progress")
            return
    
        self.view_change_in_progress = True
        new_view_number = self.view_number + 1
        last_block = self.get_last_block() if self.chain else None
    
        # Подготовка данных для запроса
        request_data = {
            'view_number': new_view_number,
            'node_id': self.node_id,
            'last_block_index': last_block.index if last_block else -1,
            'last_block_hash': last_block.hash if last_block else "0"
        }
    
        # Рассылаем запросы на смену вида другим узлам
        confirmations = 1  # Текущий узел автоматически подтверждает
        required_confirmations = (len(self.nodes) // 3 * 2) + 1  # Кворум (2f + 1)
    
        tasks = []
        for node_id, domain in self.nodes.items():
            if node_id != self.node_id:
                url = f"https://{domain}/request_view_change"
                tasks.append(self.send_post_request(node_id, url, request_data))
    
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for node_id, response in responses:
            if isinstance(response, Exception):
                node_logger.error(f"Node {node_id} failed to respond: {response}")
                continue
            status, body = response
            if status == 200 and body.get('status') == 'View change accepted':
                confirmations += 1
    
        # Если достигнут кворум, применяем изменения
        if confirmations >= required_confirmations:
            self.view_number = new_view_number
            self.is_leader = (self.node_id == new_view_number % (len(self.nodes) + 1))
            node_logger.info(f"Node {self.node_id}: View changed to {new_view_number}, is_leader={self.is_leader}")
        else:
            node_logger.warning(f"Node {self.node_id}: View change failed (confirmations: {confirmations}/{required_confirmations})")
    
        self.view_change_in_progress = False

    def shutdown(self):
        """Очищает ресурсы узла"""
        if self.leader_timeout:
            self.leader_timeout.cancel()
        if self.loop and not self.loop.is_closed():
            self.loop.close()
        node_logger.info(f"Node {self.node_id} shut down")

    async def handle_request(self, sender_id, request_data):
        """
        Обрабатывает запрос клиента, перенаправляя его лидеру, если текущий узел не лидер.
        Аргументы:
            sender_id: ID пользователя, отправившего запрос.
            request_data: Данные транзакции.
        Возвращает:
            Кортеж (success: bool, message: str) с результатом обработки.
        """
        try:
            current_app.logger.debug(f"Узел {self.node_id} обрабатывает запрос от клиента: {sender_id}")
            current_app.logger.debug(f"Данные запроса: {request_data}")
    
            # Синхронизируем view_number перед определением лидера
            await self.sync_view_number()
            current_app.logger.debug(f"Узел {self.node_id} имеет view_number: {self.view_number}")
    
            if not self.is_leader:
                # Определяем ID лидера
                leader_id = self.view_number % (len(self.nodes) + 1)
                current_app.logger.debug(f"Узел {self.node_id} перенаправляет запрос лидеру {leader_id}")
    
                if leader_id in self.nodes:
                    url = f"https://{self.nodes[leader_id]}/handle_request"
                    payload = {'sender_id': sender_id, 'request_data': request_data}
    
                    # Пытаемся перенаправить запрос с повторными попытками
                    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
                    async def send_to_leader():
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                            async with session.post(url, json=payload) as response:
                                if response.status == 200:
                                    current_app.logger.debug(f"Запрос успешно перенаправлен лидеру {leader_id}")
                                    return True, "Запрос перенаправлен лидеру"
                                else:
                                    error_text = await response.text()
                                    current_app.logger.error(f"Ошибка перенаправления лидеру {leader_id}: {error_text}")
                                    raise Exception(f"Ошибка HTTP {response.status}: {error_text}")
    
                    try:
                        success, message = await send_to_leader()
                        return success, message
                    except Exception as e:
                        current_app.logger.error(f"Не удалось перенаправить запрос лидеру {leader_id}: {str(e)}")
                        # Инициируем смену лидера
                        current_app.logger.info(f"Узел {self.node_id} инициирует смену лидера")
                        await self.initiate_view_change()
                        return False, f"Не удалось перенаправить запрос лидеру, инициирована смена лидера: {str(e)}"
                else:
                    current_app.logger.error(f"Лидер с ID {leader_id} не найден в списке узлов")
                    # Инициируем смену лидера
                    current_app.logger.info(f"Узел {self.node_id} инициирует смену лидера")
                    await self.initiate_view_change()
                    return False, "Лидер не найден, инициирована смена лидера"
    
            # Если текущий узел является лидером, обрабатываем запрос
            current_app.logger.info(f"Узел {self.node_id} является лидером, обрабатывает запрос")
            self.sequence_number += 1
            sequence_number = self.sequence_number
    
            # Добавляем user_id, timestamp и view_number в данные запроса
            request_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            request_data['user_id'] = sender_id
            request_data['view_number'] = self.view_number
            request_string = json.dumps(request_data, sort_keys=True)
            request_digest = self.generate_digest(request_string.encode('utf-8'))
    
            self.requests[sequence_number] = request_string
            current_app.logger.debug(f"Создан запрос с номером последовательности {sequence_number}")
    
            # Отправляем Pre-prepare сообщения всем узлам
            for node_id in self.nodes:
                if node_id != self.node_id:
                    await self.send_message(node_id, 'Pre-prepare', {
                        'sequence_number': sequence_number,
                        'digest': request_digest,
                        'request': request_string,
                        'view_number': self.view_number
                    })
    
            # Локально выполняем Pre-prepare
            self.pre_prepare(self.node_id, sequence_number, request_digest, request_string)
    
            # Применяем транзакцию
            success, message = await self.apply_transaction(sequence_number, request_digest)
            if not success:
                current_app.logger.error(f"Не удалось применить транзакцию: {message}")
                return False, message
    
            current_app.logger.info(f"Транзакция {sequence_number} успешно применена")
            return True, "Транзакция успешно применена"
    
        except Exception as e:
            current_app.logger.error(f"Ошибка в handle_request на узле {self.node_id}: {str(e)}", exc_info=True)
            return False, f"Неожиданная ошибка: {str(e)}"

    def pre_prepare(self, sender_id, sequence_number, digest, request):
        """Обрабатывает Pre-prepare сообщение, только если от лидера и view_number совпадает"""
        if self.is_leader or sender_id != (self.view_number % (len(self.nodes) + 1)):
            current_app.logger.warning(f"Node {self.node_id} rejected Pre-prepare from {sender_id} (not leader or wrong view)")
            return

        if sequence_number not in self.prepared:
            self.prepared[sequence_number] = {}
        if digest not in self.prepared[sequence_number]:
            self.prepared[sequence_number][digest] = set()

        self.prepared[sequence_number][digest].add(sender_id)
        self.requests[sequence_number] = request

        # Рассылаем Prepare сообщения
        for node_id in self.nodes:
            if node_id != self.node_id:
                asyncio.create_task(self.send_message(node_id, 'Prepare', {
                    'sequence_number': sequence_number,
                    'digest': digest,
                    'view_number': self.view_number
                }))

    async def prepare(self, sender_id, sequence_number, digest):
        """Обрабатывает Prepare сообщение, проверяя view_number"""
        if sequence_number not in self.prepared:
            self.prepared[sequence_number] = {}
        if digest not in self.prepared[sequence_number]:
            self.prepared[sequence_number][digest] = set()

        self.prepared[sequence_number][digest].add(sender_id)

        # Проверяем кворум (2f + 1 подтверждений, f=1 для 4 узлов)
        if len(self.prepared[sequence_number][digest]) >= 2 * 1 + 1:
            for node_id in self.nodes:
                if node_id != self.node_id:
                    await self.send_message(node_id, 'Commit', {
                        'sequence_number': sequence_number,
                        'digest': digest,
                        'view_number': self.view_number
                    })
            await self.commit(self.node_id, sequence_number, digest)

    async def commit(self, sender_id, sequence_number, digest):
        """Обрабатывает Commit сообщение, проверяя view_number"""
        if sequence_number not in self.committed:
            self.committed[sequence_number] = {}
        if digest not in self.committed[sequence_number]:
            self.committed[sequence_number][digest] = set()

        self.committed[sequence_number][digest].add(sender_id)

        # Проверяем кворум (2f + 1 подтверждений)
        if len(self.committed[sequence_number][digest]) >= 2 * 1 + 1:
            if not hasattr(self, 'applied_transactions'):
                self.applied_transactions = set()
            if sequence_number not in self.applied_transactions:
                self.applied_transactions.add(sequence_number)
                await self.apply_transaction(sequence_number, digest)

    async def receive_message(self, message):
        """Обрабатывает входящие сообщения с проверкой view_number"""
        current_app.logger.debug(f"Node {self.node_id} received: {message}")
        message_type = message['type']
        data = message['data']
        sender_id = message.get('sender_id')
        view_number = data.get('view_number', -1)

        if view_number != self.view_number:
            current_app.logger.warning(f"Node {self.node_id} rejected message with wrong view number {view_number}")
            return

        if message_type == 'Pre-prepare':
            self.pre_prepare(sender_id, data['sequence_number'], data['digest'], data['request'])
        elif message_type == 'Prepare':
            current_app.logger.debug(
                f"Node {self.node_id} is receiving prepare for sequence {data['sequence_number']} with digest {data['digest']}")
            await self.prepare(sender_id, data['sequence_number'], data['digest'])
            if data['sequence_number'] in self.prepared and data['digest'] in self.prepared[data['sequence_number']] and len(self.prepared[data['sequence_number']][data['digest']]) >= 1:
                current_app.logger.debug(
                    f"Node {self.node_id} has enough prepares for sequence {data['sequence_number']} with digest {data['digest']}")
                for node_id in self.nodes:
                    if node_id != self.node_id:
                        await self.send_message(node_id, 'Commit', {
                            'sequence_number': data['sequence_number'],
                            'digest': data['digest'],
                            'view_number': self.view_number
                        })
                await self.commit(self.node_id, data['sequence_number'], data['digest'])

    
    async def check_consensus(self, block, sender_id):
        confirmations = [sender_id]
        tasks = []
        
        for node_id, domain in self.nodes.items():
            if node_id != self.node_id and node_id != sender_id:
                url = f"https://{domain}/receive_block"
                data = {
                    'sender_id': sender_id,
                    'block': {
                        'index': block.index,
                        'timestamp': block.timestamp.isoformat(),
                        'transactions': block.transactions,
                        'previous_hash': block.previous_hash,
                        'hash': block.hash
                    }
                }
                tasks.append(self._send_block_confirmation(url, data, node_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for node_id, result in results:
            if isinstance(result, Exception):
                app.logger.error(f"Node {node_id} failed to confirm block #{block.index}: {result}")
            elif result:
                confirmations.append(node_id)
                app.logger.info(f"Node {node_id} confirmed block #{block.index}")
            else:
                app.logger.warning(f"Node {node_id} rejected block #{block.index}")
        
        return confirmations
    
    async def _send_block_confirmation(self, url, data, node_id):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return node_id, True
                    else:
                        app.logger.error(f"Node {node_id} rejected block: {await response.text()}")
                        return node_id, False
        except Exception as e:
            return node_id, e    
    
    
    genesis_lock = Lock()

    
    def create_genesis_block(self):
        with app.app_context():
            with self.genesis_lock:
                # Проверяем существование блока
                existing_genesis = db.session.query(BlockchainBlock).filter_by(index=0, node_id=self.node_id).first()
                if existing_genesis:
                    app.logger.info(f"Node {self.node_id}: Genesis block already exists")
                    return Block(
                        index=0,
                        timestamp=existing_genesis.timestamp,
                        transactions=json.loads(existing_genesis.transactions),
                        previous_hash=existing_genesis.previous_hash
                    )
                
                # Создаем новый генезис-блок с фиксированными корректными данными
                genesis_transactions = [{
                    "message": "Genesis Block",
                    "timestamp": "2025-01-01T00:00:00+00:00"
                }]
                
                # Явно задаем данные для хеширования
                genesis_data = {
                    'index': 0,
                    'timestamp': '2025-01-01T00:00:00+00:00',  # Фиксированная строка
                    'transactions': genesis_transactions,
                    'previous_hash': "0"
                }
                
                # Рассчитываем хеш
                block_string = json.dumps(genesis_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
                genesis_hash = hashlib.sha256(block_string).hexdigest()
                
                # Создаем объект блока
                genesis = Block(
                    index=0,
                    timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    transactions=genesis_transactions,
                    previous_hash="0"
                )
                genesis.hash = genesis_hash  # Устанавливаем вычисленный хеш
                
                # Сохраняем в базу данных
                genesis_db = BlockchainBlock(
                    index=genesis.index,
                    timestamp=genesis.timestamp,
                    transactions=json.dumps(genesis.transactions, ensure_ascii=False),
                    previous_hash=genesis.previous_hash,
                    hash=genesis.hash,
                    node_id=self.node_id,
                    confirming_node_id=self.node_id,
                    confirmed=True
                )
                
                try:
                    db.session.add(genesis_db)
                    db.session.commit()
                    app.logger.info(f"Node {self.node_id}: Genesis block created with hash {genesis.hash}")
                    return genesis
                except sqlalchemy.exc.IntegrityError as e:
                    db.session.rollback()
                    app.logger.warning(f"Node {self.node_id}: Failed to create genesis block due to conflict: {e}")
                    # Повторная проверка после конфликта
                    existing_genesis = db.session.query(BlockchainBlock).filter_by(index=0, node_id=self.node_id).first()
                    if existing_genesis:
                        return Block(
                            index=0,
                            timestamp=existing_genesis.timestamp,
                            transactions=json.loads(existing_genesis.transactions),
                            previous_hash=existing_genesis.previous_hash
                        )
                    raise
                return genesis
            
    async def sync_genesis_block(self):
        with app.app_context():
            # Проверяем наличие блока для текущего узла
            existing_block = db.session.query(BlockchainBlock).filter_by(
                index=0, node_id=self.node_id).first()
            if existing_block:
                app.logger.info(f"Node {self.node_id}: Genesis block exists")
                return
        
            # Пытаемся синхронизировать с другими узлами
            for node_id, domain in self.nodes.items():
                if node_id != self.node_id:
                    url = f"https://{domain}/get_block/0"
                    try:
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                            async with session.get(url) as response:
                                if response.status == 200:
                                    block_data = await response.json()
                                    with app.app_context():
                                        # Проверяем еще раз перед добавлением
                                        existing_block = db.session.query(BlockchainBlock).filter_by(
                                            index=0, node_id=self.node_id).first()
                                        if not existing_block:
                                            genesis_block = BlockchainBlock(
                                                index=block_data['index'],
                                                timestamp=datetime.fromisoformat(block_data['timestamp'].replace('Z', '+00:00')),
                                                transactions=json.dumps(block_data['transactions'], ensure_ascii=False),
                                                previous_hash=block_data['previous_hash'],
                                                hash=block_data['hash'],
                                                node_id=self.node_id,  # Используем node_id текущего узла
                                                confirming_node_id=self.node_id,
                                                confirmed=True
                                            )
                                            db.session.add(genesis_block)
                                            try:
                                                db.session.commit()
                                                app.logger.info(f"Node {self.node_id}: Genesis block synced from node {node_id}")
                                                return
                                            except sqlalchemy.exc.IntegrityError as e:
                                                db.session.rollback()
                                                app.logger.warning(f"Node {self.node_id}: Failed to sync genesis block from node {node_id}: {e}")
                                                continue
                                else:
                                    app.logger.warning(f"Node {self.node_id}: Failed to fetch genesis block from node {node_id}, status={response.status}")
                    except Exception as e:
                        app.logger.error(f"Node {self.node_id}: Failed to sync genesis from node {node_id}: {e}")
        
            # Создаем локальный блок только если синхронизация не удалась
            app.logger.info(f"Node {self.node_id}: Creating local genesis block")
            self.create_genesis_block()

    async def sync_view_number(self):
        tasks = []
        for node_id, domain in self.nodes.items():
            if node_id != self.node_id:
                url = f"https://{domain}/get_view_number"
                tasks.append(self.send_get_request(node_id, url))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        max_view_number = self.view_number
        received_views = []
        threshold = 5  # Игнорировать view_number, если разница больше 5
        
        for node_id, response in zip(self.nodes.keys(), responses):
            if isinstance(response, Exception):
                node_logger.warning(f"Failed to get view number from node {node_id}: {response}")
                continue
            try:
                status, body = response
                if status == 200 and 'view_number' in body:
                    view_number = body['view_number']
                    if view_number >= self.view_number - threshold:
                        received_views.append((node_id, view_number))
                        max_view_number = max(max_view_number, view_number)
                        node_logger.debug(f"Node {self.node_id} received view number {view_number} from node {node_id}")
                    else:
                        node_logger.warning(f"Ignored outdated view number {view_number} from node {node_id} (current: {self.view_number})")
            except ValueError as e:
                node_logger.error(f"Error unpacking response from node {node_id}: {e}, response={response}")
        
        node_logger.debug(f"Node {self.node_id} sync_view_number: received views {received_views}, max_view_number={max_view_number}")
        
        if max_view_number > self.view_number:
            self.view_number = max_view_number
            total_nodes = len(self.nodes) + 1
            self.is_leader = (self.node_id == self.view_number % total_nodes)
            node_logger.info(f"Node {self.node_id} updated view_number to {self.view_number}, is_leader={self.is_leader}")
            if not self.is_leader:
                self.start_leader_timeout()
    
    async def send_get_request(self, node_id, url):
        """Отправляет GET-запрос к указанному узлу"""
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                return response.status, await response.json()
    
    def get_last_block(self):
        return self.chain[-1]

    # В начале работы узла
    async def sync_blockchain(self):
        """Синхронизирует блокчейн текущего узла с другими узлами сети"""
        current_app.logger.info(f"Node {self.node_id} starting blockchain sync")
        
        try:
            # 1. Получаем текущую высоту локального блокчейна
            with current_app.app_context():
                local_height = db.session.query(func.max(BlockchainBlock.index)) \
                                       .filter_by(node_id=self.node_id) \
                                       .scalar() or -1
            
            # 2. Ищем самую длинную ВАЛИДНУЮ цепочку среди других узлов
            best_chain = None
            best_height = local_height
            best_node_id = None
            required_confirmations = (len(self.nodes) // 3 * 2) + 1  # Кворум
    
            for node_id, domain in self.nodes.items():
                if node_id == self.node_id:
                    continue  # Пропускаем себя
                
                try:
                    # 2.1. Запрашиваем высоту блокчейна у удаленного узла
                    async with aiohttp.ClientSession() as session:
                        # Запрос высоты
                        height_url = f"https://{domain}/get_blockchain_height"
                        async with session.get(height_url, timeout=5) as response:
                            if response.status != 200:
                                continue
                            height_data = await response.json()
                            remote_height = height_data.get('height', -1)
    
                        # Если удаленный блокчейн не длиннее, пропускаем
                        if remote_height <= best_height:
                            continue
    
                        # 2.2. Запрашиваем всю цепочку у узла
                        chain_url = f"https://{domain}/get_chain"
                        async with session.get(chain_url, timeout=10) as chain_response:
                            if chain_response.status != 200:
                                continue
                            chain_data = await chain_response.json()
                            chain_blocks = chain_data.get('chain', [])
    
                        # 2.3. Проверяем целостность всей цепочки
                        is_valid = True
                        prev_hash = "0"
                        
                        for i, block_data in enumerate(chain_blocks):
                            # Проверка последовательности индексов
                            if block_data['index'] != i:
                                is_valid = False
                                break
                                
                            # Проверка хешей
                            calculated_hash = Block.calculate_hash(block_data)
                            if block_data['hash'] != calculated_hash:
                                is_valid = False
                                break
                                
                            # Проверка связей между блоками
                            if block_data['previous_hash'] != prev_hash:
                                is_valid = False
                                break
                                
                            prev_hash = block_data['hash']
    
                        # 2.4. Проверяем количество подтверждений для каждого блока
                        if is_valid:
                            confirm_url = f"https://{domain}/get_block_details/{remote_height}"
                            async with session.get(confirm_url, timeout=5) as confirm_response:
                                if confirm_response.status == 200:
                                    confirm_data = await confirm_response.json()
                                    if confirm_data.get('confirmations', 0) >= required_confirmations:
                                        best_chain = chain_blocks
                                        best_height = remote_height
                                        best_node_id = node_id
    
                except Exception as e:
                    current_app.logger.error(f"Error syncing with node {node_id}: {str(e)}")
                    continue
    
            # 3. Если найдена более длинная валидная цепочка
            if best_chain and best_height > local_height:
                current_app.logger.info(f"Found valid longer chain (height {best_height}) from node {best_node_id}")
                
                with current_app.app_context():
                    # 3.1. Применяем все новые блоки
                    for block_data in best_chain[local_height+1:]:
                        try:
                            # Проверяем, есть ли уже такой блок
                            existing_block = BlockchainBlock.query.filter_by(
                                hash=block_data['hash'],
                                node_id=self.node_id
                            ).first()
                            
                            if not existing_block:
                                # Создаем новый блок
                                new_block = BlockchainBlock(
                                    index=block_data['index'],
                                    timestamp=datetime.fromisoformat(block_data['timestamp'].replace('Z', '+00:00')),
                                    transactions=json.dumps(block_data['transactions'], ensure_ascii=False),
                                    previous_hash=block_data['previous_hash'],
                                    hash=block_data['hash'],
                                    node_id=self.node_id,
                                    confirming_node_id=best_node_id,
                                    confirmed=True
                                )
                                db.session.add(new_block)
                                current_app.logger.debug(f"Synced block #{block_data['index']}")
                        
                        except Exception as e:
                            current_app.logger.error(f"Error processing block #{block_data['index']}: {str(e)}")
                            db.session.rollback()
                            break
                    
                    # 3.2. Фиксируем изменения
                    try:
                        db.session.commit()
                        current_app.logger.info(f"Node {self.node_id} synced to height {best_height}")
                    except Exception as e:
                        db.session.rollback()
                        current_app.logger.error(f"Commit error: {str(e)}")
    
            else:
                current_app.logger.info(f"No longer valid chain found, current height {local_height}")
    
        except Exception as e:
            current_app.logger.error(f"Critical error in blockchain sync: {str(e)}", exc_info=True)

    async def request_missing_blocks(self, from_node_id, start_index):
        """Запрашивает отсутствующие блоки у указанного узла"""
        try:
            async with aiohttp.ClientSession() as session:
                for index in range(start_index, max(0, start_index - 10), -1):  # Запрашиваем до 10 предыдущих блоков
                    url = f"https://{self.nodes[from_node_id]}/get_block/{index}"
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            block_data = await response.json()
                            # Отправляем полученный блок на обработку
                            await self.receive_message({
                                'type': 'NewBlock',
                                'sender_id': from_node_id,
                                'data': block_data
                            })
        except Exception as e:
            self.logger.error(f"Error requesting missing blocks: {str(e)}")
    
    async def request_block_from_node(self, node_id, block_index):
        """Запросить блок с определенным индексом у узла"""
        try:
            node = self.nodes[node_id]
            host = node.host
            port = node.port

            async with aiohttp.ClientSession() as session:
                url = f"https://{host}:{port}/get_block/{block_index}"
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        block_data = await response.json()

                        # Проверяем и сохраняем блок
                        with app.app_context():
                            existing_block = BlockchainBlock.query.filter_by(hash=block_data['hash']).first()
                            if not existing_block:
                                new_block_db = BlockchainBlock(
                                    index=block_data['index'],
                                    timestamp=datetime.datetime.fromisoformat(block_data['timestamp']),
                                    transactions=json.dumps(block_data['transactions'], ensure_ascii=False),
                                    previous_hash=block_data['previous_hash'],
                                    hash=block_data['hash'],
                                    node_id=block_data['node_id']
                                )
                                db.session.add(new_block_db)
                                db.session.commit()
                                app.logger.info(f"Synced block #{block_index} from node {node_id}")
        except Exception as e:
            app.logger.error(f"Error requesting block {block_index} from node {node_id}: {e}")

    async def send_message(self, node_id, message_type, data):
        message = {
            'type': message_type,
            'data': data,
            'sender_id': self.node_id
        }
        url = f"https://{self.nodes[node_id]}/receive_message"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.post(url, json=message) as response:
                    app.logger.debug(f"Sent message to node {node_id}: {message_type}")
                    return response.status == 200
        except Exception as e:
            app.logger.error(f"Exception in send_message to node {node_id}: {e}")
            return False

    async def send_post_request(self, node_id, url, payload):
        try:
            if not isinstance(payload, dict):
                app.logger.error(f"Invalid payload type for node {node_id}: {type(payload)}")
                return node_id, Exception("Invalid payload type")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(url, json=payload) as response:
                    status = response.status
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' not in content_type:
                        body = await response.text()
                        app.logger.error(f"Node {node_id} returned non-JSON response: status={status}, content-type={content_type}, body={body[:500]}")
                        return node_id, (status, {"error": "Non-JSON response", "body": body[:500]})
                    body = await response.json()
                    return node_id, (status, body)
        except Exception as e:
            app.logger.error(f"Error sending POST to node {node_id} at {url}: {e}", exc_info=True)
            return node_id, e
    

    async def broadcast_new_block(self, block, transaction_record, block_db):
        start_time = asyncio.get_event_loop().time()  # Начало замера времени
        total_nodes = len(self.nodes) + 1
        f = (total_nodes - 1) // 3
        required_confirmations = 2 * f + 1
        confirmations = 1  # Считаем текущий узел
        
        block_dict = {
            'index': block.index,
            'timestamp': block.timestamp.isoformat(),
            'transactions': block.transactions,
            'previous_hash': block.previous_hash,
            'hash': block.hash
        }
        current_app.logger.debug(f"Broadcasting block #{block.index} with hash {block.hash}")
        
        tasks = []
        for node_id, domain in self.nodes.items():
            if node_id != self.node_id:
                url = f"https://{domain}/receive_block"
                payload = {
                    "sender_id": self.node_id,
                    "block": block_dict
                }
                tasks.append(self.send_post_request(node_id, url, payload))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for node_id, response in responses:
            if isinstance(response, Exception):
                current_app.logger.error(f"Error sending to node {node_id}: {response}")
                continue
            try:
                status, body = response
                current_app.logger.debug(f"Node {node_id} response: status={status}, body={body}")
                if status == 200 and body.get("status") in ["Block accepted", "Block already exists"]:
                    confirmations += 1
                    current_app.logger.info(f"Node {node_id} confirmed block #{block.index}")
                else:
                    current_app.logger.error(f"Node {node_id} failed to confirm block #{block.index}: {body}")
            except Exception as e:
                current_app.logger.error(f"Error processing response from node {node_id}: {e}")
        
        end_time = asyncio.get_event_loop().time()  # Конец замера времени
        consensus_time = end_time - start_time
        self.consensus_times.append(consensus_time)
        if len(self.consensus_times) > 100:  # Ограничиваем размер списка
            self.consensus_times.pop(0)
        
        current_app.logger.info(f"Consensus check: {confirmations}/{required_confirmations} confirmations, time: {consensus_time:.2f} sec")
        if confirmations >= required_confirmations:
            try:
                with current_app.app_context():
                    existing_block = db.session.query(BlockchainBlock).filter_by(
                        hash=block.hash,
                        node_id=self.node_id
                    ).first()
                    if existing_block:
                        if not existing_block.confirmed:
                            existing_block.confirmed = True
                            db.session.commit()
                            current_app.logger.info(f"Block #{block.index} already exists, updated confirmation")
                    else:
                        db.session.add(transaction_record)
                        db.session.add(block_db)
                        block_db.confirmed = True
                        db.session.commit()
                        current_app.logger.info(f"Block #{block.index} committed")
            except Exception as e:
                db.session.rollback()
                current_app.logger.error(f"Database error committing block #{block.index}: {e}", exc_info=True)
                return confirmations, total_nodes
        else:
            current_app.logger.warning(f"Consensus not reached for block #{block.index}: {confirmations}/{required_confirmations}")
            db.session.rollback()
        
        return confirmations, total_nodes

    # Проверка доступности узла
    async def check_node_availability(self, node_id):
        if node_id == self.node_id:
            return True
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://{self.nodes[node_id].host}/health"
                async with session.get(url, timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            app.logger.error(f"Error checking node {node_id} availability: {e}")
            return False

    async def check_nodes_status(self):
        """Проверяет статус всех узлов, включая текущий"""
        status = {}
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        
        # Собираем задачи для проверки высоты цепочки и view_number всех узлов
        tasks = []
        for node_id, domain in self.nodes.items():
            if node_id != self.node_id:
                url_height = f"https://{domain}/get_blockchain_height"
                url_view = f"https://{domain}/get_view_number"
                tasks.append((node_id, self.send_get_request(node_id, url_height)))
                tasks.append((node_id, self.send_get_request(node_id, url_view)))
        
        # Проверяем текущий узел
        current_node_id = self.node_id
        status[current_node_id] = {
            'is_online': True,  # Изначально предполагаем, что текущий узел онлайн
            'block_count': None,
            'view_number': self.view_number,
            'is_leader': self.is_leader,
            'host': self.host,
            'port': self.port
        }
        
        # Проверяем доступность текущего узла через внутренний запрос
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                url = f"https://{self.host}:{self.port}/health"
                async with session.get(url, timeout=5) as response:
                    status[current_node_id]['is_online'] = (response.status == 200)
        except Exception as e:
            node_logger.error(f"Current node {current_node_id} is offline: {e}")
            status[current_node_id]['is_online'] = False
        
        # Выполняем запросы к другим узлам
        responses = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        
        # Обрабатываем результаты
        for i, (node_id, _) in enumerate(tasks):
            response = responses[i]
            if node_id not in status:
                status[node_id] = {
                    'is_online': False,
                    'block_count': None,
                    'view_number': None,
                    'is_leader': False,
                    'host': self.nodes[node_id].split(':')[0] if ':' in self.nodes[node_id] else self.nodes[node_id],
                    'port': self.nodes[node_id].split(':')[1] if ':' in self.nodes[node_id] else '443'
                }
            
            if isinstance(response, Exception):
                node_logger.warning(f"Failed to get data from node {node_id}: {response}")
                continue
            
            try:
                status_code, body = response
                if status_code == 200:
                    status[node_id]['is_online'] = True
                    if 'height' in body:
                        status[node_id]['block_count'] = body['height']
                    if 'view_number' in body:
                        status[node_id]['view_number'] = body['view_number']
                        # Определяем лидера
                        total_nodes = len(self.nodes) + 1
                        status[node_id]['is_leader'] = (node_id == body['view_number'] % total_nodes)
                else:
                    node_logger.warning(f"Invalid response from node {node_id}: status={status_code}, body={body}")
            except Exception as e:
                node_logger.error(f"Error processing response from node {node_id}: {e}")
        
        # Получаем количество блоков для текущего узла из базы
        with current_app.app_context():
            status[current_node_id]['block_count'] = db.session.query(BlockchainBlock).filter_by(node_id=current_node_id).count()
        
        node_logger.debug(f"Nodes status: {status}")
        return status

    def get_network_stats(self):
        """Возвращает статистику сети"""
        avg_consensus_time = sum(self.consensus_times) / len(self.consensus_times) if self.consensus_times else 0
        tps = len(self.consensus_times) / (sum(self.consensus_times) or 1)  # Транзакций в секунду
        view_change_success_rate = (sum(self.view_change_success) / len(self.view_change_success) * 100) if self.view_change_success else 0
        
        return {
            'avg_consensus_time': round(avg_consensus_time, 2),
            'tps': round(tps, 2),
            'view_change_success_rate': round(view_change_success_rate, 2),
            'consensus_times': self.consensus_times[-10:],  # Последние 10 значений для графика
            'view_change_success': self.view_change_success[-10:]  # Последние 10 результатов
        }
    
    async def apply_transaction(self, sequence_number, digest):
        app.logger.debug(f"Applying transaction {sequence_number} with digest {digest}")
    
        request = self.requests.get(sequence_number)
        if not request:
            app.logger.error(f"Request with sequence number {sequence_number} not found.")
            return False, "Request with sequence number not found."
    
        try:
            transaction_data = json.loads(request)
            app.logger.debug(f"Transaction data to apply: {transaction_data}")
    
            with app.app_context():
                try:
                    # Проверка обязательных полей
                    required_fields = ['ДокументID', 'Единица_ИзмеренияID', 'Количество',
                                       'СкладОтправительID', 'СкладПолучательID', 'ТоварID', 'user_id']
                    for field in required_fields:
                        if field not in transaction_data:
                            return False, f"Missing required field: {field}"
    
                    user_id = transaction_data['user_id']
                    if not user_id:
                        return False, "User ID cannot be empty"
    
                    # Нормализация временной метки
                    timestamp = transaction_data.get('timestamp')
                    if isinstance(timestamp, str):
                        try:
                            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            normalized_timestamp = timestamp
                        except ValueError:
                            normalized_timestamp = datetime.now(timezone.utc).isoformat()
                    elif hasattr(timestamp, 'isoformat'):
                        normalized_timestamp = timestamp.isoformat()
                    else:
                        normalized_timestamp = datetime.now(timezone.utc).isoformat()
    
                    # Подготовка данных для хеширования
                    transaction_for_hash = {
                        'ДокументID': int(transaction_data['ДокументID']),
                        'Единица_ИзмеренияID': int(transaction_data['Единица_ИзмеренияID']),
                        'Количество': float(transaction_data['Количество']),
                        'СкладОтправительID': int(transaction_data['СкладОтправительID']),
                        'СкладПолучательID': int(transaction_data['СкладПолучательID']),
                        'ТоварID': int(transaction_data['ТоварID']),
                        'user_id': int(user_id),
                        'timestamp': normalized_timestamp
                    }
    
                    # Генерация хеша транзакции
                    transaction_string = json.dumps(
                        transaction_for_hash,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(',', ':')
                    )
                    transaction_hash = hashlib.sha256(transaction_string.encode('utf-8')).hexdigest()
                    app.logger.info(f"Transaction hash generated: {transaction_hash}")
    
                    # Обновление запасов
                    success, message = update_запасы(
                        transaction_data['СкладПолучательID'],
                        transaction_data['ТоварID'],
                        transaction_data['Количество']
                    )
                    if not success:
                        return False, message
    
                    # Обработка расхода между разными складами
                    if transaction_data['СкладОтправительID'] != transaction_data['СкладПолучательID']:
                        success, message = update_запасы(
                            transaction_data['СкладОтправительID'],
                            transaction_data['ТоварID'],
                            -transaction_data['Количество']
                        )
                        if not success:
                            return False, message
    
                    # Создание записи о транзакции
                    new_record = ПриходРасход(
                        СкладОтправительID=transaction_data['СкладОтправительID'],
                        СкладПолучательID=transaction_data['СкладПолучательID'],
                        ДокументID=transaction_data['ДокументID'],
                        ТоварID=transaction_data['ТоварID'],
                        Количество=transaction_data['Количество'],
                        Единица_ИзмеренияID=transaction_data['Единица_ИзмеренияID'],
                        TransactionHash=transaction_hash,
                        Timestamp=normalized_timestamp,
                        user_id=user_id
                    )
    
                    # Проверка актуального последнего блока
                    last_block = BlockchainBlock.query.order_by(BlockchainBlock.index.desc()).first()
                    expected_index = last_block.index + 1 if last_block else 0
    
                    # Создание нового блока
                    new_block = Block(
                        index=expected_index,
                        timestamp=datetime.now(timezone.utc),
                        transactions=[transaction_data],
                        previous_hash=last_block.hash if last_block else '0' * 64
                    )
    
                    # Подготовка объекта блока для базы
                    block_db = BlockchainBlock(
                        index=new_block.index,
                        timestamp=new_block.timestamp,
                        transactions=json.dumps(new_block.transactions, ensure_ascii=False),
                        previous_hash=new_block.previous_hash,
                        hash=new_block.hash,
                        node_id=self.node_id,
                        confirming_node_id=self.node_id,
                        confirmed=False  # Изначально неподтвержден
                    )
    
                    # Рассылка блока другим узлам, передаем объекты для сохранения
                    confirmations, total_nodes = await self.broadcast_new_block(new_block, new_record, block_db)
    
                    if confirmations >= ((total_nodes - 1) // 3 * 2) + 1:
                        app.logger.info(f"Consensus reached for block #{new_block.index}")
                        return True, "Transaction applied successfully"
                    else:
                        app.logger.warning(f"Consensus not reached for block #{new_block.index}")
                        db.session.rollback()
                        return False, "Consensus not reached"
    
                except Exception as db_error:
                    db.session.rollback()
                    app.logger.error(f"Database error: {str(db_error)}", exc_info=True)
                    return False, f"Database error: {str(db_error)}"
    
        except json.JSONDecodeError as json_error:
            app.logger.error(f"JSON decode error: {str(json_error)}")
            return False, f"Invalid transaction data: {str(json_error)}"
        except Exception as e:
            app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return False, f"Unexpected error: {str(e)}"

    def generate_digest(self, message):
        return hashlib.sha256(message).hexdigest()

    def calculate_hash(self):
        data = str(self.index) + str(self.timestamp) + json.dumps(self.transactions, sort_keys=True) + str(self.previous_hash)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    @staticmethod
    def verify_block_integrity(block):
        try:
            if not block:
                return False, "Блок не существует"
            
            # Для генезис-блока применяем особые правила проверки
            if block.index == 0:
                expected_data = {
                    'index': 0,
                    'timestamp': '2025-01-01T00:00:00+00:00',
                    'transactions': [{
                        "message": "Genesis Block",
                        "timestamp": "2025-01-01T00:00:00+00:00"
                    }],
                    'previous_hash': "0"
                }
                
                # Рассчитываем ожидаемый хеш
                expected_hash = hashlib.sha256(
                    json.dumps(expected_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
                ).hexdigest()
                
                if block.hash != expected_hash:
                    return False, (
                        f"Неверный хеш генезис-блока. Ожидалось: {expected_hash}, "
                        f"получено: {block.hash}. Генезис-блок должен иметь фиксированные данные."
                    )
                
                return True, "Генезис-блок достоверен"
            
            # Проверка для обычных блоков (остается без изменений)
            normalized_timestamp = block.timestamp.isoformat() if block.timestamp else None
            if block.timestamp and not block.timestamp.tzinfo:
                normalized_timestamp = datetime.fromtimestamp(block.timestamp.timestamp(), tz=timezone.utc).isoformat()
    
            block_data = {
                'index': block.index,
                'timestamp': normalized_timestamp,
                'transactions': json.loads(block.transactions) if block.transactions else [],
                'previous_hash': block.previous_hash
            }
            calculated_hash = hashlib.sha256(
                json.dumps(block_data, sort_keys=True).encode('utf-8')
            ).hexdigest()
            
            if calculated_hash != block.hash:
                return False, f"Хэш блока не совпадает с вычисленным (ожидалось: {calculated_hash}, получено: {block.hash})"
    
            if block.index > 0:
                prev_block = BlockchainBlock.query.filter_by(
                    index=block.index - 1,
                    node_id=block.node_id
                ).first()
                
                if not prev_block:
                    return False, "Предыдущий блок не найден"
                    
                if block.previous_hash != prev_block.hash:
                    return False, f"Хэш предыдущего блока не совпадает (ожидалось: {prev_block.hash}, получено: {block.previous_hash})"
            
            confirmations = BlockchainBlock.query.filter_by(
                index=block.index,
                hash=block.hash
            ).all()
            
            total_nodes = len(NODE_DOMAINS)
            required_confirmations = (total_nodes - 1) // 3 * 2 + 1
            
            if len(confirmations) < required_confirmations:
                return False, f"Недостаточно подтверждений ({len(confirmations)} из {required_confirmations})"
    
            return True, "Блок достоверен"
            
        except Exception as e:
            return False, f"Ошибка при проверке блока: {str(e)}"

# Создаем узлы блокчейна
# Используем динамическое создание на основе переменных окружения:
NODE_ID = int(os.environ.get('NODE_ID', 0))
PORT = int(os.environ.get('PORT', 5000))
DOMAIN_PREFIX = os.environ.get('DOMAIN_PREFIX', '')


# Создаем узлы блокчейна
nodes = {}
for i in range(4):
    nodes[i] = Node(
        node_id=i,
        nodes={j: NODE_DOMAINS[j] for j in range(4) if j != i},  # Используем строки доменов
        host=NODE_DOMAINS[i],
        port=443
    )
current_node = nodes[NODE_ID]

def serialize_data(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True)

async def check_node_availability(self, node_id):
    if node_id == self.node_id:
        return True
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://{self.nodes[node_id].host}/health"
            async with session.get(url, timeout=5) as response:
                return response.status == 200
    except Exception as e:
        app.logger.error(f"Error checking node {node_id} availability: {e}")
        return False

# Маршрут receive_block
@app.route('/receive_block', methods=['POST'])
@csrf.exempt
async def receive_block():
    """Обрабатывает получение нового блока от других узлов сети"""
    try:
        # 1. Парсинг и валидация входящих данных
        data = request.get_json()
        if not data or 'sender_id' not in data or 'block' not in data:
            app.logger.error("Invalid block data: missing required fields")
            return jsonify({'error': 'Invalid request: missing sender_id or block data'}), 400

        sender_id = data['sender_id']
        block_data = data['block']
        
        # 2. Проверка формата данных блока
        required_fields = ['index', 'timestamp', 'transactions', 'previous_hash', 'hash']
        if not all(field in block_data for field in required_fields):
            app.logger.error(f"Invalid block format: missing fields in {block_data}")
            return jsonify({'error': 'Invalid block format'}), 400

        # 3. Получаем текущий узел
        current_node = nodes.get(NODE_ID)
        if not current_node:
            app.logger.error(f"Current node {NODE_ID} not found")
            return jsonify({'error': 'Node not initialized'}), 500

        # 4. Проверяем, существует ли блок уже в нашей базе
        with app.app_context():
            existing_block = db.session.query(BlockchainBlock).filter_by(
                hash=block_data['hash'],
                node_id=NODE_ID
            ).first()

            if existing_block:
                app.logger.info(f"Block #{block_data['index']} already exists")
                return jsonify({'status': 'Block already exists'}), 200

            # 5. Проверяем целостность блока
            try:
                # 5.1. Проверяем хеш блока
                calculated_hash = Block.calculate_hash(block_data)
                if calculated_hash != block_data['hash']:
                    app.logger.error(f"Hash mismatch for block #{block_data['index']}: "
                                   f"calculated {calculated_hash}, received {block_data['hash']}")
                    return jsonify({'error': 'Block hash mismatch'}), 400

                # 5.2. Для не-генезис блоков проверяем предыдущий блок
                if block_data['index'] > 0:
                    prev_block = db.session.query(BlockchainBlock).filter_by(
                        index=block_data['index'] - 1,
                        node_id=NODE_ID
                    ).order_by(BlockchainBlock.timestamp.desc()).first()

                    if not prev_block:
                        # Если предыдущего блока нет - запрашиваем его
                        app.logger.warning(f"Previous block #{block_data['index']-1} not found, requesting...")
                        await current_node.request_missing_blocks(sender_id, block_data['index'] - 1)
                        return jsonify({'status': 'Requested missing blocks'}), 202

                    if block_data['previous_hash'] != prev_block.hash:
                        app.logger.error(f"Previous hash mismatch for block #{block_data['index']}: "
                                       f"expected {prev_block.hash}, got {block_data['previous_hash']}")
                        return jsonify({'error': 'Previous hash mismatch'}), 400

                # 6. Создаем объект блока
                try:
                    block = Block(
                        index=block_data['index'],
                        timestamp=datetime.fromisoformat(block_data['timestamp'].replace('Z', '+00:00')),
                        transactions=block_data['transactions'],
                        previous_hash=block_data['previous_hash']
                    )
                    block.hash = block_data['hash']  # Используем полученный хеш
                except Exception as e:
                    app.logger.error(f"Failed to create block object: {str(e)}")
                    return jsonify({'error': 'Invalid block data'}), 400

                # 7. Сохраняем блок в базу данных
                block_db = BlockchainBlock(
                    index=block.index,
                    timestamp=block.timestamp,
                    transactions=json.dumps(block.transactions, ensure_ascii=False),
                    previous_hash=block.previous_hash,
                    hash=block.hash,
                    node_id=NODE_ID,
                    confirming_node_id=sender_id,
                    confirmed=False  # Ждем подтверждения консенсуса
                )

                db.session.add(block_db)

                # 8. Проверяем консенсус (если блок от лидера)
                if sender_id == current_node.view_number % (len(current_node.nodes) + 1):
                    confirmations = 1  # Учитываем текущий узел
                    required_confirmations = (len(current_node.nodes) // 3 * 2) + 1

                    # Проверяем подтверждения от других узлов
                    for node_id, domain in current_node.nodes.items():
                        if node_id != NODE_ID and node_id != sender_id:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    url = f"https://{domain}/confirm_block/{block.index}"
                                    async with session.get(url, timeout=5) as response:
                                        if response.status == 200:
                                            confirmations += 1
                            except Exception:
                                continue

                    if confirmations >= required_confirmations:
                        block_db.confirmed = True
                        app.logger.info(f"Block #{block.index} confirmed by {confirmations} nodes")
                    else:
                        app.logger.warning(f"Block #{block.index} has only {confirmations}/{required_confirmations} confirmations")

                db.session.commit()
                app.logger.info(f"Block #{block.index} saved successfully")

                # 9. Если это новый блок, распространяем его дальше
                if block.index > current_node.get_last_block().index:
                    for node_id, domain in current_node.nodes.items():
                        if node_id != NODE_ID and node_id != sender_id:
                            asyncio.create_task(current_node.send_message(
                                node_id,
                                'NewBlock',
                                {'block': block_data}
                            ))

                return jsonify({'status': 'Block accepted'}), 200

            except sqlalchemy.exc.IntegrityError as e:
                db.session.rollback()
                app.logger.error(f"Database integrity error: {str(e)}")
                return jsonify({'error': 'Database error'}), 500
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Error processing block: {str(e)}", exc_info=True)
                return jsonify({'error': str(e)}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in receive_block: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


async def broadcast_confirmation(block_index, creator_node_id, confirming_node_id):
    """Рассылает подтверждение блока другим узлам"""
    try:
        block = BlockchainBlock.query.filter_by(
            index=block_index,
            node_id=creator_node_id,
            confirming_node_id=confirming_node_id
        ).first()
        
        if not block:
            return

        block_data = {
            'index': block.index,
            'timestamp': block.timestamp.isoformat(),
            'transactions': json.loads(block.transactions),
            'previous_hash': block.previous_hash,
            'hash': block.hash,
            'node_id': block.node_id
        }

        for node_id, node in nodes.items():
            if node_id != NODE_ID and node_id != creator_node_id:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://{node.host}:{node.port}/receive_confirmation"
                        payload = {
                            'block': block_data,
                            'confirming_node_id': confirming_node_id
                        }
                        await session.post(url, json=payload, timeout=5)
                except Exception as e:
                    app.logger.error(f"Error broadcasting confirmation to node {node_id}: {e}")

    except Exception as e:
        app.logger.error(f"Error in broadcast_confirmation: {e}")

@app.route('/inventory_report')
@login_required
def inventory_report():
    try:
        # Получаем все данные по запасам
        inventory_data = db.session.query(
            Запасы, Склады, Товары, Единица_измерения
        ).join(
            Склады, Запасы.СкладID == Склады.СкладID
        ).join(
            Товары, Запасы.ТоварID == Товары.ТоварID
        ).join(
            Единица_измерения, Запасы.Единица_ИзмеренияID == Единица_измерения.Единица_ИзмеренияID
        ).all()

        # Суммарные запасы по товарам по всем складам
        total_by_product = {}
        for item in inventory_data:
            запас, склад, товар, ед_измерения = item

            if товар.ТоварID not in total_by_product:
                total_by_product[товар.ТоварID] = {
                    'product_name': товар.Наименование,
                    'total_quantity': 0,
                    'unit': ед_измерения.Единица_Измерения
                }

            total_by_product[товар.ТоварID]['total_quantity'] += запас.Количество

        # Суммарные запасы по складам
        total_by_warehouse = {}
        for item in inventory_data:
            запас, склад, товар, ед_измерения = item

            if склад.СкладID not in total_by_warehouse:
                total_by_warehouse[склад.СкладID] = {
                    'warehouse_name': склад.Название,
                    'products': {},
                    'total_products': 0
                }

            total_by_warehouse[склад.СкладID]['products'][товар.ТоварID] = {
                'product_name': товар.Наименование,
                'quantity': запас.Количество,
                'unit': ед_измерения.Единица_Измерения
            }

            total_by_warehouse[склад.СкладID]['total_products'] += 1

        # Добавляем алиасы для складов
        SenderWarehouse = db.aliased(Склады)
        ReceiverWarehouse = db.aliased(Склады)
        
        # История движения товаров (последние 10 транзакций)
        latest_transactions = db.session.query(
            ПриходРасход,
            SenderWarehouse.Название.label('sender_warehouse'),
            ReceiverWarehouse.Название.label('receiver_warehouse'),
            Товары.Наименование.label('product_name'),
            Тип_документа.Тип_документа.label('document_type'),
            Единица_измерения.Единица_Измерения.label('unit')
        ).join(
            Товары, ПриходРасход.ТоварID == Товары.ТоварID
        ).join(
            Тип_документа, ПриходРасход.ДокументID == Тип_документа.ДокументID
        ).join(
            Единица_измерения, ПриходРасход.Единица_ИзмеренияID == Единица_измерения.Единица_ИзмеренияID
        ).join(
            SenderWarehouse, ПриходРасход.СкладОтправительID == SenderWarehouse.СкладID  # Алиас отправителя
        ).join(
            ReceiverWarehouse, ПриходРасход.СкладПолучательID == ReceiverWarehouse.СкладID  # Алиас получателя
        ).order_by(
            ПриходРасход.Timestamp.desc()
        ).limit(10).all()

        # Склады с нулевыми запасами по определенным товарам
        warehouses = Склады.query.all()
        products = Товары.query.all()

        zero_inventory = []
        for warehouse in warehouses:
            for product in products:
                stock = Запасы.query.filter_by(
                    СкладID=warehouse.СкладID,
                    ТоварID=product.ТоварID
                ).first()

                if not stock:
                    zero_inventory.append({
                        'warehouse_name': warehouse.Название,
                        'product_name': product.Наименование
                    })
                elif stock.Количество == 0:
                    zero_inventory.append({
                        'warehouse_name': warehouse.Название,
                        'product_name': product.Наименование,
                        'last_update': stock.Дата_обновления.strftime('%d.%m.%Y')
                    })

        # Преобразуем timestamp в datetime если это строка
        for trans in latest_transactions:
            if isinstance(trans.ПриходРасход.Timestamp, str):
                trans.ПриходРасход.Timestamp = datetime.fromisoformat(trans.ПриходРасход.Timestamp.replace('Z', '+00:00'))
        
        current_time = datetime.now(timezone.utc)
        
        return render_template(
            'inventory_report.html',
            inventory_data=inventory_data,
            total_by_product=total_by_product,
            total_by_warehouse=total_by_warehouse,
            latest_transactions=latest_transactions,
            zero_inventory=zero_inventory,
            current_time=current_time  # Добавляем текущее время
        )

    except Exception as e:
        app.logger.error(f"Ошибка при формировании отчета по запасам: {e}")
        flash(f"Ошибка при формировании отчета: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/test_transaction', methods=['GET'])
@login_required
async def test_transaction():
    try:
        # Тестовая транзакция
        transaction_data = {
            'СкладОтправительID': 3,
            'СкладПолучательID': 3,
            'ДокументID': 2,  # Приходная накладная
            'ТоварID': 1,
            'Количество': 10.0,
            'Единица_ИзмеренияID': 3,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': current_user.id  # Добавляем ID текущего пользователя
        }

        # Получаем текущий узел
        node_id = 0
        if current_user.role == 'admin':
            node_id = 0
        elif current_user.role == 'north':
            node_id = 1
        elif current_user.role == 'south':
            node_id = 2
        elif current_user.role == 'west':
            node_id = 3

        # Применяем транзакцию напрямую
        app.logger.debug(f"Testing direct transaction application on node {node_id}")

        # Проверка наличия товара и склада в БД
        with app.app_context():
            склад = Склады.query.get(transaction_data['СкладОтправительID'])
            товар = Товары.query.get(transaction_data['ТоварID'])

            if not склад:
                flash(f"Ошибка: Склад с ID {transaction_data['СкладОтправительID']} не найден", 'danger')
                return redirect(url_for('index'))

            if not товар:
                flash(f"Ошибка: Товар с ID {transaction_data['ТоварID']} не найден", 'danger')
                return redirect(url_for('index'))

        # Генерируем данные для транзакции
        request_string = json.dumps(transaction_data, sort_keys=True)
        request_digest = nodes[node_id].generate_digest(request_string.encode('utf-8'))

        # Сохраняем запрос
        sequence_number = nodes[node_id].sequence_number + 1
        nodes[node_id].sequence_number = sequence_number
        nodes[node_id].requests[sequence_number] = request_string

        # Применяем транзакцию
        success, message = await nodes[node_id].apply_transaction(sequence_number, request_digest)

        if success:
            flash('Тестовая транзакция успешно применена!', 'success')
        else:
            flash(f'Ошибка при применении тестовой транзакции: {message}', 'danger')

        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f"Error in test_transaction: {e}")
        flash(f'Ошибка: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.template_filter('pprint')
def pprint_filter(data):
    if isinstance(data, dict):
        return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    return str(data)

def check_node_availability_sync(node_id):
    """Синхронная проверка доступности узла"""
    if node_id == NODE_ID:  # Текущий узел всегда доступен
        return True
    
    try:
        response = requests.get(
            f'https://{nodes[node_id].host}/health',
            timeout=2
        )
        return response.status_code == 200
    except Exception:
        return False

@app.route('/blockchain')
@login_required
def view_blockchain():
    """Отображает всю цепочку блоков с информацией о подтверждениях"""
    try:
        # Получаем все уникальные индексы блоков
        block_indices = db.session.query(
            BlockchainBlock.index
        ).distinct().order_by(BlockchainBlock.index).all()

        formatted_blocks = []
        total_nodes = len(nodes) + 1  # Учитываем текущий узел

        for index in block_indices:
            index = index[0]
            
            # Получаем все подтверждения для этого блока (без фильтра по node_id)
            confirmations = BlockchainBlock.query.filter_by(index=index).all()
            
            if not confirmations:
                continue

            # Группируем по хэшу блока, чтобы обрабатывать блоки с одинаковым содержимым
            blocks_by_hash = {}
            for conf in confirmations:
                if conf.hash not in blocks_by_hash:
                    blocks_by_hash[conf.hash] = []
                blocks_by_hash[conf.hash].append(conf)

            # Для каждого уникального хэша создаем запись
            for block_hash, blocks in blocks_by_hash.items():
                main_block = blocks[0]
                
                try:
                    transactions = json.loads(main_block.transactions)
                    
                    # Получаем список уникальных узлов, подтвердивших этот блок
                    confirming_nodes = list({b.confirming_node_id for b in blocks if b.confirmed})
                    confirmations_count = len(confirming_nodes)
                    
                    # Проверяем достижение консенсуса
                    required_confirmations = (total_nodes // 3 * 2) + 1
                    consensus_reached = confirmations_count >= required_confirmations
                    
                    # Обновляем статус подтверждения в БД, если консенсус достигнут
                    if consensus_reached and not all(b.confirmed for b in blocks):
                        for b in blocks:
                            b.confirmed = True
                        db.session.commit()
                        app.logger.debug(f"Updated confirmed status for block #{index} with hash {block_hash}")

                    formatted_transactions = []
                    for transaction in transactions:
                        formatted_transaction = {}
                        for key, value in transaction.items():
                            if key == 'СкладОтправительID':
                                склад = Склады.query.get(value)
                                formatted_transaction['СкладОтправитель'] = f"{склад.Название} (ID: {value})" if склад else f"Склад (ID: {value})"
                            elif key == 'СкладПолучательID':
                                склад = Склады.query.get(value)
                                formatted_transaction['СкладПолучатель'] = f"{склад.Название} (ID: {value})" if склад else f"Склад (ID: {value})"
                            elif key == 'ДокументID':
                                doc = Тип_документа.query.get(value)
                                formatted_transaction['Документ'] = f"{doc.Тип_документа} (ID: {value})" if doc else f"Документ (ID: {value})"
                            elif key == 'ТоварID':
                                товар = Товары.query.get(value)
                                formatted_transaction['Товар'] = f"{товар.Наименование} (ID: {value})" if товар else f"Товар (ID: {value})"
                            elif key == 'Единица_ИзмеренияID':
                                unit = Единица_измерения.query.get(value)
                                formatted_transaction['Единица_Измерения'] = f"{unit.Единица_Измерения} (ID: {value})" if unit else f"Ед. изм. (ID: {value})"
                            else:
                                formatted_transaction[key] = value
                        formatted_transactions.append(formatted_transaction)
                    
                    formatted_blocks.append({
                        'index': index,
                        'timestamp': main_block.timestamp,
                        'transactions': formatted_transactions,
                        'previous_hash': main_block.previous_hash,
                        'hash': main_block.hash,
                        'node_id': main_block.node_id,
                        'is_genesis': index == 0,
                        'confirmations': confirmations_count,
                        'total_nodes': total_nodes,
                        'confirming_nodes': confirming_nodes,
                        'consensus_reached': consensus_reached
                    })
                except json.JSONDecodeError as e:
                    app.logger.error(f"Error decoding transactions for block {index}: {e}")
                    continue

        app.logger.debug(f"Loaded {len(formatted_blocks)} blocks from database")
        return render_template('blockchain.html', blocks=formatted_blocks)
    except Exception as e:
        app.logger.error(f"Error in view_blockchain: {e}")
        flash(f'Ошибка при загрузке блокчейна: {e}', 'danger')
        return redirect(url_for('index'))

@app.route('/receive_message', methods=['POST'])
@csrf.exempt
async def receive_message():
    """Обрабатывает входящие сообщения от других узлов."""
    message = request.get_json()
    if not message or not isinstance(message, dict):
        return jsonify({"error": "Invalid message format"}), 400
    if 'sender_id' not in message or 'block' not in message:
        return jsonify({"error": "Missing required fields in message"}), 400
    
    # Здесь продолжается логика обработки сообщения, например:
    sender_id = message['sender_id']
    block = message['block']
    # ... (остальная логика обработки блока, например, его валидация и добавление в цепочку)
    
    return jsonify({"status": "Message received"}), 200

@app.route('/register', methods=['GET', 'POST'])
def register():
    app.logger.debug('Entering register function')
    form = RegistrationForm()
    if form.validate_on_submit():
        invitation = Invitation.query.filter_by(code=form.invitation_code.data, user_id=None).first()
        if not invitation:
            flash('Неверный код приглашения.', 'danger')
            return render_template('register.html', form=form)

        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password, role=form.role.data)
        db.session.add(new_user)
        db.session.commit()

        invitation.user_id = new_user.id
        invitation.used_at = datetime.datetime.now(timezone.utc)
        db.session.commit()

        flash('Регистрация прошла успешно! Теперь вы можете войти.', 'success')
        app.logger.debug('Exiting register function')
        return redirect(url_for('login'))
    app.logger.debug('Exiting register function')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    app.logger.debug('Entering login function')
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            app.logger.debug('Exiting login function')
            return redirect(next_page or url_for('index'))
        else:
            flash('Неверное имя пользователя или пароль', 'danger')
            app.logger.debug('Exiting login function with error')
            return render_template('login.html', form=form, error='Неверное имя пользователя или пароль')
    app.logger.debug('Exiting login function')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    app.logger.debug('Entering logout function')
    logout_user()
    app.logger.debug('Exiting logout function')
    return redirect(url_for('login'))


@app.route('/check_inventory')
@login_required
def check_inventory_query():
    try:
        warehouse_id = request.args.get('warehouse_id', type=int)
        item_id = request.args.get('item_id', type=int)

        if not warehouse_id or not item_id:
            return jsonify({'success': False, 'message': 'Не указан склад или товар'}), 400

        inventory = Запасы.query.filter_by(
            СкладID=warehouse_id,
            ТоварID=item_id
        ).first()

        item = Товары.query.get(item_id)
        unit = Единица_измерения.query.get(item.Единица_ИзмеренияID) if item else None

        if inventory:
            return jsonify({
                'success': True,
                'available': inventory.Количество,
                'unit': unit.Единица_Измерения if unit else '',
                'last_update': inventory.Дата_обновления.isoformat() if inventory.Дата_обновления else None
            })
        else:
            return jsonify({
                'success': True,
                'available': 0,
                'unit': unit.Единица_Измерения if unit else ''
            })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/debug_inventory')
@login_required
def debug_inventory():
    if not current_user.is_admin:
        flash('Доступ запрещен', 'danger')
        return redirect(url_for('index'))

    try:
        # Получаем все записи о запасах
        all_inventory = Запасы.query.all()

        # Форматируем данные для отображения
        inventory_data = []
        for inv in all_inventory:
            try:
                warehouse = Склады.query.get(inv.СкладID)
                item = Товары.query.get(inv.ТоварID)
                unit = Единица_измерения.query.get(inv.Единица_ИзмеренияID) if hasattr(inv,
                                                                                       'Единица_ИзмеренияID') and inv.Единица_ИзмеренияID else None

                inventory_data.append({
                    'склад_id': inv.СкладID,
                    'склад': warehouse.Название if warehouse else 'Неизвестный склад',
                    'товар_id': inv.ТоварID,
                    'товар': item.Наименование if item else 'Неизвестный товар',
                    'количество': inv.Количество,
                    'единица': unit.Единица_Измерения if unit else 'Неизвестно'
                })
            except Exception as item_error:
                app.logger.error(f"Ошибка при обработке записи запаса: {item_error}")
                inventory_data.append({
                    'склад_id': inv.СкладID,
                    'товар_id': inv.ТоварID,
                    'ошибка': str(item_error)
                })

        # Отобразим отладочную информацию на странице
        return render_template('debug_inventory.html', inventory_data=inventory_data, count=len(all_inventory))
    except Exception as e:
        app.logger.error(f"Ошибка при отладке запасов: {e}")
        flash(f"Ошибка при отладке запасов: {e}", 'danger')
        return redirect(url_for('index'))

# Обновляем маршрут index для отправки запроса лидеру
@app.route('/', methods=['GET', 'POST'])
@login_required
async def index():
    current_app.logger.debug('Entering index function')
    connection_status = ""

    try:
        db.session.execute(text("SELECT 1"))
        connection_status = "Соединение с базой данных установлено!"
        current_app.logger.info(connection_status)
    except Exception as e:
        connection_status = f"Ошибка подключения к базе данных: {e}"
        current_app.logger.error(connection_status)
        flash(connection_status, 'danger')

    form = PrihodRashodForm()
    transaction_data = None

    try:
        prihod_rashod_records = ПриходРасход.query.order_by(ПриходРасход.ПриходРасходID.desc()).all()
        current_inventory = Запасы.query.all()
        current_app.logger.debug(f"Загружено {len(prihod_rashod_records)} записей из базы данных")
    except Exception as db_error:
        error_msg = f"Ошибка при загрузке записей из базы данных: {db_error}"
        current_app.logger.error(error_msg)
        flash(error_msg, 'danger')
        prihod_rashod_records = []
        current_inventory = []

    if request.method == 'POST':
        current_app.logger.debug(f"POST-запрос получен: {request.form}")
        is_ajax_request = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        if not form.validate():
            current_app.logger.error(f"Ошибки валидации формы: {form.errors}")
            if is_ajax_request:
                return jsonify({
                    'success': False,
                    'message': f"Ошибка валидации формы: {form.errors}"
                }), 400
            else:
                flash(f"Ошибка валидации формы: {form.errors}", 'danger')
                return render_template('index.html', form=form, records=prihod_rashod_records,
                                       inventory=current_inventory,
                                       connection_status=connection_status, transaction_data=transaction_data)

        if form.validate_on_submit():
            current_app.logger.debug("Форма успешно прошла валидацию")

            try:
                transaction_data = {
                    'СкладОтправительID': form.СкладОтправительID.data,
                    'СкладПолучательID': form.СкладПолучательID.data,
                    'ДокументID': form.ДокументID.data,
                    'ТоварID': form.ТоварID.data,
                    'Количество': form.Количество.data,
                    'Единица_ИзмеренияID': form.Единица_ИзмеренияID.data,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

                current_app.logger.debug(f"Сформированы данные транзакции: {transaction_data}")

                node_id = None
                if current_user.role == 'admin':
                    node_id = 0
                elif current_user.role == 'north':
                    node_id = 1
                elif current_user.role == 'south':
                    node_id = 2
                elif current_user.role == 'west':
                    node_id = 3
                else:
                    error_msg = f"Неподдерживаемая роль пользователя: {current_user.role}"
                    current_app.logger.error(error_msg)
                    if is_ajax_request:
                        return jsonify({'success': False, 'message': error_msg}), 400
                    else:
                        flash(error_msg, 'danger')
                        return redirect(url_for('index'))

                current_app.logger.debug(f"Выбран узел {node_id} для обработки транзакции")

                if node_id not in nodes:
                    error_msg = f"Узел с ID {node_id} не найден"
                    current_app.logger.error(error_msg)
                    if is_ajax_request:
                        return jsonify({'success': False, 'message': error_msg}), 400
                    else:
                        flash(error_msg, 'danger')
                        return redirect(url_for('index'))

                # Отправляем запрос на обработку текущему узлу, который перенаправит его лидеру
                success, message = await nodes[node_id].handle_request(current_user.id, transaction_data)
                if success:
                    current_app.logger.debug(f"Транзакция успешно отправлена на узел {node_id}")
                    if is_ajax_request:
                        return jsonify({'success': True, 'message': 'Запись успешно добавлена'}), 200
                    else:
                        flash('Запись успешно добавлена', 'success')
                else:
                    current_app.logger.error(f"Ошибка при обработке запроса узлом {node_id}: {message}")
                    if is_ajax_request:
                        return jsonify(
                            {'success': False, 'message': f"Ошибка при добавлении записи: {message}"}), 400
                    else:
                        flash(f"Ошибка при добавлении записи: {message}", 'danger')

            except Exception as e:
                error_msg = f"Ошибка при подготовке транзакции: {e}"
                current_app.logger.error(error_msg)
                if is_ajax_request:
                    return jsonify({'success': False, 'message': error_msg}), 500
                else:
                    flash(error_msg, 'danger')

            if not is_ajax_request:
                return redirect(url_for('index'))

    current_app.logger.debug('Exiting index function')
    return render_template('index.html', form=form, records=prihod_rashod_records,
                           inventory=current_inventory,
                           connection_status=connection_status, transaction_data=transaction_data)


# Новый маршрут для получения последней записи
@app.route('/get_last_record')
@login_required
def get_last_record():
    try:
        last_record = ПриходРасход.query.order_by(ПриходРасход.ПриходРасходID.desc()).first()
        if last_record:
            return jsonify({
                'success': True,
                'record': {
                    'ПриходРасходID': last_record.ПриходРасходID,
                    'СкладОтправитель': last_record.СкладОтправитель.Название,
                    'СкладПолучатель': last_record.СкладПолучатель.Название,
                    'Тип_документа': last_record.Тип_документа.Тип_документа,
                    'Товары': last_record.Товары.Наименование,
                    'Количество': last_record.Количество,
                    'Единица_измерения': last_record.Единица_измерения.Единица_Измерения,
                    'TransactionHash': last_record.TransactionHash
                }
            })
        return jsonify({'success': False, 'message': 'Записи не найдены'})
    except Exception as e:
        app.logger.error(f"Ошибка при получении последней записи: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin/invitations', methods=['GET'])
@login_required
def admin_invitations():
    app.logger.debug('Entering admin_invitations function')
    if not current_user.is_admin:
        flash('У вас нет прав для доступа к этой странице', 'danger')
        app.logger.debug('Exiting admin_invitations function with error')
        return redirect(url_for('index'))
    form = InvitationForm()
    invitations = Invitation.query.all()
    app.logger.debug('Exiting admin_invitations function')
    return render_template('admin_invitations.html', form=form, invitations=invitations)

@app.route('/admin/create_invitation', methods=['POST'])
@login_required
def admin_create_invitation():
    app.logger.debug('Entering admin_create_invitation function')
    if not current_user.is_admin:
        flash('У вас нет прав для доступа к этой странице', 'danger')
        app.logger.debug('Exiting admin_create_invitation function with error')
        return redirect(url_for('index'))
    form = InvitationForm()
    if form.validate_on_submit():
        code = str(uuid4())
        new_invitation = Invitation(code=code, email=form.email.data)
        db.session.add(new_invitation)
        db.session.commit()
        flash(f'Приглашение успешно создано. Код: {code}', 'success')
        app.logger.debug('Exiting admin_create_invitation function')
        return redirect(url_for('admin_invitations'))
    app.logger.debug('Exiting admin_create_invitation function')
    return render_template('admin_invitations.html', form=form, invitations=Invitation.query.all())


def check_data_integrity(record_id, transaction_data=None):
    """Проверяет целостность данных транзакции, включая временные метки"""
    try:
        # Получаем запись из базы данных
        record = ПриходРасход.query.get(record_id)
        if not record:
            app.logger.error(f"Record with ID {record_id} not found")
            return {
                'success': False,
                'message': "Запись не найдена",
                'details': f"Запись с ID {record_id} не существует в базе данных"
            }

        # Проверяем наличие хэша транзакции
        if not record.TransactionHash:
            app.logger.warning(f"Record {record_id} has no transaction hash")
            return {
                'success': False,
                'message': "Отсутствует хэш транзакции",
                'details': "Запись не была подтверждена в блокчейне"
            }

        # Нормализация временной метки из записи
        def get_normalized_timestamp(ts):
            if ts is None:
                return None
            if isinstance(ts, str):
                try:
                    # Пробуем распарсить строку в datetime
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    return dt.isoformat()
                except ValueError:
                    # Если не получается распарсить, возвращаем как есть
                    return ts
            elif hasattr(ts, 'isoformat'):
                return ts.isoformat()
            return str(ts)

        record_timestamp = get_normalized_timestamp(record.Timestamp)

        # Формируем данные для проверки
        check_data = {
            'ДокументID': int(record.ДокументID),
            'Единица_ИзмеренияID': int(record.Единица_ИзмеренияID),
            'Количество': float(record.Количество),
            'СкладОтправительID': int(record.СкладОтправительID),
            'СкладПолучательID': int(record.СкладПолучательID),
            'ТоварID': int(record.ТоварID),
            'user_id': int(record.user_id),
            'timestamp': record_timestamp  # Используем нормализованную метку
        }

        # Генерируем хэш для проверки
        check_string = json.dumps(
            check_data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(',', ':')
        )
        computed_hash = hashlib.sha256(check_string.encode('utf-8')).hexdigest()

        # Сравниваем хэши
        if computed_hash == record.TransactionHash:
            app.logger.info(f"Integrity check passed for record {record_id}")
            return {
                'success': True,
                'message': "Целостность данных подтверждена",
                'details': None,
                'computed_hash': computed_hash,
                'stored_hash': record.TransactionHash,
                'timestamp_used': record_timestamp
            }

        # Дополнительная проверка: возможно timestamp был в другом формате
        alternative_hashes = []
        if record_timestamp:
            # Вариант 1: Заменяем 'T' на пробел
            alt_timestamp1 = record_timestamp.replace('T', ' ')
            alt_data1 = {**check_data, 'timestamp': alt_timestamp1}
            alt_hash1 = hashlib.sha256(
                json.dumps(alt_data1, sort_keys=True, ensure_ascii=False).encode('utf-8')
            ).hexdigest()
            alternative_hashes.append(alt_hash1)

            # Вариант 2: Удаляем микросекунды
            if '.' in record_timestamp:
                alt_timestamp2 = record_timestamp.split('.')[0] + 'Z'
                alt_data2 = {**check_data, 'timestamp': alt_timestamp2}
                alt_hash2 = hashlib.sha256(
                    json.dumps(alt_data2, sort_keys=True, ensure_ascii=False).encode('utf-8')
                ).hexdigest()
                alternative_hashes.append(alt_hash2)

        # Проверяем альтернативные варианты хэшей
        for alt_hash in alternative_hashes:
            if alt_hash == record.TransactionHash:
                app.logger.info(f"Integrity check passed with alternative timestamp format for record {record_id}")
                return {
                    'success': True,
                    'message': "Целостность подтверждена (альтернативный формат времени)",
                    'details': "Хэш совпал при использовании альтернативного формата временной метки",
                    'computed_hash': alt_hash,
                    'stored_hash': record.TransactionHash,
                    'timestamp_used': alt_timestamp1 if alt_hash == alt_hash1 else alt_timestamp2
                }

        # Если ни один вариант не подошел
        app.logger.error(f"Integrity check failed for record {record_id}")
        return {
            'success': False,
            'message': "Обнаружены расхождения в данных",
            'details': (
                f"Вычисленный хэш ({computed_hash}) не совпадает с сохраненным ({record.TransactionHash}).\n"
                f"Использованные данные: {check_data}"
            ),
            'computed_hash': computed_hash,
            'stored_hash': record.TransactionHash,
            'timestamp_used': record_timestamp,
            'transaction_data': check_data
        }

    except Exception as e:
        app.logger.error(f"Error in integrity check for record {record_id}: {str(e)}", exc_info=True)
        return {
            'success': False,
            'message': "Ошибка при проверке целостности",
            'details': str(e),
            'error_type': type(e).__name__
        }


@app.route('/check_integrity/<int:record_id>', methods=['POST'])
@login_required
def check_integrity_route(record_id):
    app.logger.debug('Entering check_integrity_route function')
    data = request.get_json()
    transaction_data = data.get('transaction_data') if data else None
    result = check_data_integrity(record_id, transaction_data)
    app.logger.debug('Exiting check_integrity_route function')
    return jsonify(result)


def update_запасы(склад_id, товар_id, количество):
    with app.app_context():
        try:
            # Получаем информацию о складе и товаре для улучшения сообщений об ошибках
            склад = Склады.query.get(склад_id)
            товар = Товары.query.get(товар_id)

            # Проверяем, существуют ли склад и товар
            if not склад:
                return False, f"Склад с ID {склад_id} не найден"
            if not товар:
                return False, f"Товар с ID {товар_id} не найден"
            # Для расходных операций проверяем остаток на складе
            if количество < 0:
                запас = Запасы.query.filter_by(СкладID=склад_id, ТоварID=товар_id).first()
                if not запас:
                    return False, (f"Невозможно отгрузить {abs(количество)} "
                                   f"единиц товара '{товар.Наименование}' со склада '{склад.Название}', "
                                   f"так как товар отсутствует на складе")

                if запас.Количество + количество < 0:
                    return False, (f"Недостаточно товара '{товар.Наименование}' "
                                   f"на складе '{склад.Название}'. Требуется: {abs(количество)}, в наличии: {запас.Количество}")
            запас = Запасы.query.filter_by(СкладID=склад_id, ТоварID=товар_id).first()
            if запас:
                # Обновляем существующую запись
                запас.Количество += количество
                запас.Дата_обновления = datetime.now(timezone.utc).date()
                db.session.commit()
                операция = "приход" if количество > 0 else "расход"
                return True, (f"Обновлены запасы товара '{товар.Наименование}' "
                              f"на складе '{склад.Название}'. {операция}: {abs(количество)}, новый остаток: {запас.Количество}")
            else:
                # Создаем новую запись
                # Для расхода уже проверено выше, что такой ситуации не может быть
                единица_измерения = товар.Единица_ИзмеренияID
                new_запас = Запасы(
                    СкладID=склад_id,
                    ТоварID=товар_id,
                    Количество=количество,
                    Дата_обновления=datetime.now(timezone.utc).date(),
                    Единица_ИзмеренияID=единица_измерения  # Используем единицу измерения из товара
                )
                db.session.add(new_запас)
                db.session.commit()
                return True, (f"Создана запись о товаре '{товар.Наименование}' "
                              f"на складе '{склад.Название}'. Начальный остаток: {количество}")
        except Exception as e:
            db.session.rollback()
            error_message = f"Ошибка при обновлении запасов: {e}"
            app.logger.error(error_message)
            return False, error_message


@app.route('/get_record_details/<int:record_id>')
@login_required
def get_record_details(record_id):
    try:
        # Создаем явные алиасы для складов
        SenderWarehouse = db.aliased(Склады)
        ReceiverWarehouse = db.aliased(Склады)

        record = db.session.query(
            ПриходРасход,
            SenderWarehouse.Название.label('sender_name'),
            ReceiverWarehouse.Название.label('receiver_name'),
            Товары.Наименование,
            Тип_документа.Тип_документа,
            Единица_измерения.Единица_Измерения,
            User.username,
            User.role,
            ПриходРасход.TransactionHash,
            ПриходРасход.Timestamp
        ).join(
            SenderWarehouse, ПриходРасход.СкладОтправительID == SenderWarehouse.СкладID
        ).join(
            ReceiverWarehouse, ПриходРасход.СкладПолучательID == ReceiverWarehouse.СкладID
        ).join(
            Товары, ПриходРасход.ТоварID == Товары.ТоварID
        ).join(
            Тип_документа, ПриходРасход.ДокументID == Тип_документа.ДокументID
        ).join(
            Единица_измерения, ПриходРасход.Единица_ИзмеренияID == Единица_измерения.Единица_ИзмеренияID
        ).join(
            User, ПриходРасход.user_id == User.id
        ).filter(
            ПриходРасход.ПриходРасходID == record_id
        ).first()

        if not record:
            return jsonify({'success': False, 'message': 'Запись не найдена'}), 404

        (record_data, sender_name, receiver_name, item_name,
         doc_type, unit, username, role, tx_hash, timestamp) = record

        # Форматируем дату
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    op_date = datetime.fromisoformat(timestamp).strftime('%d.%m.%Y %H:%M')
                except ValueError:
                    op_date = timestamp
            else:
                op_date = timestamp.strftime('%d.%m.%Y %H:%M')
        else:
            op_date = 'Не указана'

        # Определяем узел блокчейна
        node_mapping = {'admin': 0, 'north': 1, 'south': 2, 'west' : 3}
        node_id = node_mapping.get(role.lower(), None)
        node_info = f'Узел #{node_id} ({role})' if node_id is not None else f'Узел не определен (роль: {role})'

        return jsonify({
            'success': True,
            'data': {
                'record_id': record_data.ПриходРасходID,
                'sender_warehouse': sender_name or 'Не указан',
                'receiver_warehouse': receiver_name or 'Не указан',
                'document_type': doc_type or 'Не указан',
                'item_name': item_name or 'Не указан',
                'quantity': float(record_data.Количество),
                'unit': unit or 'Не указана',
                'operation_date': op_date,
                'transaction_hash': tx_hash,
                'user_name': username or 'Неизвестный',
                'user_role': role or 'Не указана',
                'node_info': node_info
            }
        })
    except Exception as e:
        app.logger.error(f"Error in get_record_details: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/confirm_in_blockchain/<int:record_id>', methods=['POST'])
@login_required
async def confirm_in_blockchain(record_id):  # Добавили async здесь
    try:
        record = ПриходРасход.query.get(record_id)
        if not record:
            return jsonify({'success': False, 'message': 'Запись не найдена'}), 404

        if record.TransactionHash:
            return jsonify({'success': False, 'message': 'Запись уже подтверждена в блокчейне'}), 400

        transaction_data = {
            'record_id': record.ПриходРасходID,
            'sender_warehouse': record.СкладОтправительID,
            'receiver_warehouse': record.СкладПолучательID,
            'item_id': record.ТоварID,
            'quantity': float(record.Количество),
            'timestamp': record.Timestamp.isoformat() if record.Timestamp else None,
            'user_id': current_user.id
        }

        node_id = None
        if current_user.role == 'admin':
            node_id = 0
        elif current_user.role == 'north':
            node_id = 1
        elif current_user.role == 'south':
            node_id = 2
        elif current_user.role == 'west':
            node_id = 3

        if node_id is None:
            node_info = f'Узел #{node_id} ({user_role})' if node_id is not None else 'Не указан'

        # Получаем экземпляр узла
        node = nodes.get(node_id)
        if not node:
            return jsonify({'success': False, 'message': 'Узел блокчейна не найден'}), 500

        # Используем await, так как функция теперь async
        success, message = await node.handle_request(current_user.id, transaction_data)

        if success:
            # Обновляем запись в базе данных
            record.TransactionHash = hashlib.sha256(
                json.dumps(transaction_data, sort_keys=True).encode('utf-8')
            ).hexdigest()
            db.session.commit()
            return jsonify({'success': True, 'message': 'Запись подтверждена в блокчейне'})
        else:
            return jsonify({'success': False, 'message': message}), 500

    except Exception as e:
        app.logger.error(f"Error confirming in blockchain: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get_unit_of_measurement/<int:item_id>', methods=['GET'])
def get_unit_of_measurement(item_id):
    товар = Товары.query.get(item_id)
    if товар:
        единица_измерения = Единица_измерения.query.get(товар.Единица_ИзмеренияID)
        if единица_измерения:
            return jsonify({'Единица_ИзмеренияID': единица_измерения.Единица_ИзмеренияID,
                            'Единица_Измерения': единица_измерения.Единица_Измерения})
    return jsonify({'error': 'Товар или единица измерения не найдены'}), 404


@app.route('/get_item_info/<int:item_id>')
def get_item_info(item_id):
    try:
        item = Товары.query.get(item_id)
        if not item:
            return jsonify({'success': False, 'message': 'Товар не найден'})

        unit = Единица_измерения.query.get(item.Единица_ИзмеренияID)

        # Формируем путь к изображению
        image_url = url_for('static', filename=f'img/products/{item.image_path}') if item.image_path else None

        return jsonify({
            'success': True,
            'item_id': item.ТоварID,
            'item_name': item.Наименование,
            'description': item.Описание if hasattr(item, 'Описание') else 'Не указано',
            'unit': unit.Единица_Измерения if unit else 'не указано',
            'unit_id': item.Единица_ИзмеренияID,
            'image_url': image_url  # Добавляем URL изображения
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/nodes_status')
@login_required
async def nodes_status():
    node = nodes.get(NODE_ID)
    if not node:
        app.logger.error(f"Node {NODE_ID} not found")
        return jsonify({'error': 'Node not found'}), 404
    nodes_status = await node.check_nodes_status()
    nodes_info = [
        {
            'node_id': node_id,
            'host': info['host'],
            'port': info['port'],
            'block_count': info['block_count'],
            'is_online': info['is_online'],
            'is_leader': info['is_leader']
        }
        for node_id, info in nodes_status.items()
    ]
    network_stats = node.get_network_stats()
    return render_template('nodes_status.html', nodes=nodes_info, network_stats=network_stats)

@app.route('/get_block_details/<int:block_index>')
def get_block_details(block_index):
    try:
        # Получаем все блоки с этим индексом
        blocks = BlockchainBlock.query.filter_by(index=block_index).all()
        if not blocks:
            return jsonify({'error': 'Block not found'}), 404

        # Берем первый блок как основной
        main_block = blocks[0]
        transactions = json.loads(main_block.transactions) if main_block.transactions else []
        
        # Получаем список уникальных узлов, подтвердивших этот блок
        confirming_nodes = list({b.confirming_node_id for b in blocks})
        
        return jsonify({
            'index': main_block.index,
            'timestamp': main_block.timestamp.isoformat(),
            'transactions': transactions,
            'hash': main_block.hash,
            'previous_hash': main_block.previous_hash,
            'node_id': main_block.node_id,
            'tx_count': len(transactions),
            'confirmations': len(confirming_nodes),
            'total_nodes': len(nodes),
            'confirming_nodes': confirming_nodes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/receive_confirmation', methods=['POST'])

async def receive_confirmation():
    """Обработчик для приема подтверждений блоков от других узлов"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        block = data['block']
        confirming_node_id = data['confirming_node_id']

        # Проверяем, существует ли уже такое подтверждение
        existing = BlockchainBlock.query.filter_by(
            index=block['index'],
            node_id=block['node_id'],
            confirming_node_id=confirming_node_id
        ).first()

        if not existing:
            new_confirmation = BlockchainBlock(
                index=block['index'],
                timestamp=datetime.fromisoformat(block['timestamp']),
                transactions=json.dumps(block['transactions']),
                previous_hash=block['previous_hash'],
                hash=block['hash'],
                node_id=block['node_id'],
                confirming_node_id=confirming_node_id,
                confirmed=False
            )
            db.session.add(new_confirmation)
            db.session.commit()

        return jsonify({"status": "Confirmation accepted"}), 200

    except Exception as e:
        app.logger.error(f"Error processing confirmation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_chain', methods=['GET'])
async def get_chain():
    with app.app_context():
        chain = BlockchainBlock.query.filter_by(confirmed=True).order_by(BlockchainBlock.index.asc()).all()
        chain_data = [
            {
                'index': block.index,
                'timestamp': block.timestamp.isoformat(),
                'transactions': json.loads(block.transactions),
                'previous_hash': block.previous_hash,
                'hash': block.hash,
                'node_id': block.node_id,
                'confirming_node_id': block.confirming_node_id
            } for block in chain
        ]
        return jsonify({'chain': chain_data})

@app.route('/debug_blockchain')
@login_required
def debug_blockchain():
    blocks = db.session.query(BlockchainBlock).all()
    block_data = [
        {
            'index': b.index,
            'hash': b.hash,
            'node_id': b.node_id,
            'confirming_node_id': b.confirming_node_id,
            'confirmed': b.confirmed
        } for b in blocks
    ]
    return jsonify({'blocks': block_data})

@app.route('/verify_block/<int:block_index>')
@login_required
def verify_block(block_index):
    try:
        blocks = BlockchainBlock.query.filter_by(index=block_index).all()
        if not blocks:
            return jsonify({
                'success': False,
                'message': f'Блок с индексом {block_index} не найден',
                'block_index': block_index
            }), 404
        
        results = []
        for block in blocks:
            try:
                is_valid, message = Node.verify_block_integrity(block)  # Изменено на вызов статического метода
                results.append({
                    'block_index': block.index,
                    'node_id': block.node_id,
                    'is_valid': is_valid,
                    'message': message,
                    'hash': block.hash,
                    'previous_hash': block.previous_hash,
                    'confirmations': BlockchainBlock.query.filter_by(
                        index=block.index,
                        hash=block.hash
                    ).count()
                })
            except Exception as block_error:
                app.logger.error(f"Error verifying block {block_index}: {str(block_error)}")
                results.append({
                    'block_index': block_index,
                    'node_id': block.node_id if block else None,
                    'is_valid': False,
                    'message': f"Ошибка проверки: {str(block_error)}",
                    'hash': block.hash if block else None,
                    'previous_hash': block.previous_hash if block else None,
                    'confirmations': 0
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'block_index': block_index
        })
    
    except Exception as e:
        app.logger.error(f"Error in verify_block endpoint for block {block_index}: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f"Внутренняя ошибка сервера при проверке блока {block_index}",
            'error': str(e),
            'block_index': block_index
        }), 500

@app.route('/test_verify_block/<int:block_index>')
def test_verify_block(block_index):
    try:
        block = BlockchainBlock.query.filter_by(index=block_index).first()
        if not block:
            return jsonify({'error': 'Block not found'}), 404
            
        return jsonify({
            'index': block.index,
            'hash': block.hash,
            'exists': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

async def start_sync(node):
    await node.sync_blockchain()

# Новый маршрут для обработки запросов смены вида
@app.route('/request_view_change', methods=['POST'])
@csrf.exempt
async def request_view_change():
    """Обрабатывает запрос на смену вида (view change)"""
    try:
        data = request.get_json()
        if not data or 'view_number' not in data or 'node_id' not in data:
            current_app.logger.error("Invalid view change request format")
            return jsonify({'success': False, 'message': 'Invalid request format'}), 400

        node = nodes.get(NODE_ID)
        if not node:
            current_app.logger.error(f"Node {NODE_ID} not found")
            return jsonify({'success': False, 'message': 'Node not found'}), 404

        proposed_view = data['view_number']
        if proposed_view <= node.view_number:
            current_app.logger.warning(f"Rejected view change: proposed {proposed_view} <= current {node.view_number}")
            return jsonify({'success': False, 'message': 'Proposed view number is too low'}), 400

        # Проверяем целостность цепочки, если данные предоставлены
        if 'last_block_index' in data and 'last_block_hash' in data:
            last_block = node.get_last_block()
            if last_block and (data['last_block_index'] != last_block.index or 
                              data['last_block_hash'] != last_block.hash):
                current_app.logger.warning("Chain mismatch detected during view change")
                return jsonify({'success': False, 'message': 'Chain mismatch'}), 400

        # Обновляем view_number и статус лидера
        node.view_number = proposed_view
        total_nodes = len(node.nodes) + 1
        node.is_leader = (node.node_id == proposed_view % total_nodes)

        current_app.logger.info(f"View changed to {proposed_view}, is_leader={node.is_leader}")
        
        if not node.is_leader:
            node.start_leader_timeout()

        return jsonify({
            'status': 'View change accepted',
            'view_number': node.view_number,
            'is_leader': node.is_leader
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error processing view change: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 500

# Новый маршрут для подтверждения смены вида
@app.route('/receive_view_change', methods=['POST'])
@csrf.exempt
async def receive_view_change():
    """Обрабатывает подтверждение смены вида от других узлов"""
    data = request.get_json()
    if not data or 'view_number' not in data or 'node_id' not in data:
        current_app.logger.error("Invalid view change confirmation format")
        return jsonify({'success': False, 'message': 'Invalid view change confirmation format'}), 400

    node = nodes.get(NODE_ID)
    if not node:
        current_app.logger.error(f"Node {NODE_ID} not found")
        return jsonify({'success': False, 'message': 'Node not found'}), 404

    new_view_number = data['view_number']
    if new_view_number <= node.view_number:
        current_app.logger.warning(f"Node {NODE_ID} rejected view change confirmation: proposed view {new_view_number} <= current view {node.view_number}")
        return jsonify({'success': False, 'message': 'Proposed view number is too low'}), 400

    # Проверяем целостность цепочки
    last_block = node.get_last_block() if node.chain else None
    if last_block:
        if data['last_block_index'] > last_block.index:
            current_app.logger.warning(f"Node {NODE_ID} has outdated chain: local index {last_block.index}, received index {data['last_block_index']}")
            # Запрашиваем синхронизацию цепочки
            await node.sync_blockchain()
        elif data['last_block_hash'] != last_block.hash:
            current_app.logger.error(f"Node {NODE_ID} detected chain mismatch: local hash {last_block.hash}, received hash {data['last_block_hash']}")
            return jsonify({'success': False, 'message': 'Chain mismatch'}), 400

    node.view_number = new_view_number
    node.is_leader = (node.node_id == new_view_number % (len(node.nodes) + 1))
    current_app.logger.info(f"Node {NODE_ID} confirmed view change to {new_view_number}, is_leader={node.is_leader}")
    if not node.is_leader:
        node.start_leader_timeout()
    return jsonify({'status': 'View change confirmed'}), 200

@app.route('/confirm_block/<int:block_index>', methods=['GET'])
@csrf.exempt
async def confirm_block(block_index):
    """Подтверждает наличие блока на этом узле"""
    with app.app_context():
        exists = db.session.query(BlockchainBlock.query.filter_by(
            index=block_index,
            node_id=NODE_ID
        ).exists()).scalar()
        
        return jsonify({'exists': exists}), 200 if exists else 404

@app.route('/get_view_number', methods=['GET'])
@csrf.exempt  # Отключаем CSRF, так как это системный маршрут между узлами
async def get_view_number():
    """Возвращает текущий номер вида (view_number) для текущего узла"""
    node = nodes.get(NODE_ID)
    if not node:
        node_logger.error(f"Node {NODE_ID} not found")
        return jsonify({'success': False, 'message': 'Node not found'}), 404

    node_logger.debug(f"Node {NODE_ID} returning view_number: {node.view_number}")
    return jsonify({'view_number': node.view_number}), 200

# Новый маршрут для обработки запросов лидером
@app.route('/handle_request', methods=['POST'])
@csrf.exempt
async def handle_leader_request():
    """Обрабатывает запрос, перенаправленный нелидеровым узлом"""
    data = request.get_json()
    if not data or 'sender_id' not in data or 'request_data' not in data:
        current_app.logger.error("Invalid request format")
        return jsonify({'success': False, 'message': 'Invalid request format'}), 400

    node = nodes.get(NODE_ID)
    if not node:
        current_app.logger.error(f"Node {NODE_ID} not found")
        return jsonify({'success': False, 'message': 'Node not found'}), 404

    if not node.is_leader:
        current_app.logger.error(f"Node {NODE_ID} is not the leader")
        return jsonify({'success': False, 'message': 'This node is not the leader'}), 400

    success, message = await node.handle_request(data['sender_id'], data['request_data'])
    if success:
        return jsonify({'success': True, 'message': message}), 200
    else:
        return jsonify({'success': False, 'message': message}), 400

def cleanup():
    """Очищает ресурсы всех узлов при остановке приложения"""
    for node_id, node in nodes.items():
        node.shutdown()
    node_logger.info("All nodes shut down")

atexit.register(cleanup)

if __name__ == '__main__':
    current_node = nodes[NODE_ID]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_sync(current_node))
    app.run(host='0.0.0.0', port=PORT)
