import os
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
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
from sqlalchemy import text, select
from flask_wtf.csrf import CSRFProtect
import threading
import datetime
import psycopg2
import asyncio
import aiohttp
from uuid import uuid4
from flask_cors import CORS
from flask_session import Session
import redis

load_dotenv()

app = Flask(__name__)

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

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_default_secret_key')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=int(older_than))
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
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=int(older_than))
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

    def calculate_hash(self):
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp.isoformat(),
            'transactions': self.transactions,
            'previous_hash': self.previous_hash
        }, sort_keys=True).encode('utf-8')
        return hashlib.sha256(block_string).hexdigest()

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

class Node:
    def __init__(self, node_id, nodes, host, port):
        self.node_id = node_id
        self.nodes = nodes
        self.host = host
        self.port = port
        self.sequence_number = 0
        self.prepared = {}
        self.committed = {}
        self.log = []
        self.is_leader = (node_id == 0)
        self.requests = {}
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, datetime.datetime.now(), [], "0")

    def get_last_block(self):
        return self.chain[-1]

    # В начале работы узла
    async def sync_blockchain(self):
        """Синхронизировать блокчейн с другими узлами при запуске"""
        app.logger.info(f"Node {self.node_id} начинает синхронизацию блокчейна")

        # Получаем последний локальный блок
        with app.app_context():
            local_last_block = BlockchainBlock.query.order_by(BlockchainBlock.index.desc()).first()
            local_index = local_last_block.index if local_last_block else -1

        # Запрашиваем у других узлов их последние блоки
        for node_id, node in self.nodes.items():
            if node_id != self.node_id:
                try:
                    host = node.host
                    port = node.port

                    async with aiohttp.ClientSession() as session:
                        url = f"http://{host}:{port}/get_blockchain_height"
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                remote_height = data.get('height', -1)

                                # Если у удаленного узла блокчейн длиннее
                                if remote_height > local_index:
                                    # Запрашиваем недостающие блоки
                                    for i in range(local_index + 1, remote_height + 1):
                                        await self.request_block_from_node(node_id, i)
                except Exception as e:
                    app.logger.error(f"Error during blockchain sync with node {node_id}: {e}")

        app.logger.info(f"Node {self.node_id} закончил синхронизацию блокчейна")

    async def request_block_from_node(self, node_id, block_index):
        """Запросить блок с определенным индексом у узла"""
        try:
            node = self.nodes[node_id]
            host = node.host
            port = node.port

            async with aiohttp.ClientSession() as session:
                url = f"http://{host}:{port}/get_block/{block_index}"
                async with session.get(url) as response:
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

    async def send_message(self, recipient_id, message_type, data):
        try:
            recipient = self.nodes.get(recipient_id)
            if not recipient:
                app.logger.error(f"Recipient node {recipient_id} not found in nodes list")
                return False

            host = recipient.get('host', 'localhost')
            port = recipient.get('port', 5000)

            async with aiohttp.ClientSession() as session:
                url = f"http://{host}:{port}/node_message"
                payload = {
                    'sender_id': self.node_id,
                    'message_type': message_type,
                    'data': data
                }
                app.logger.debug(f"Sending {message_type} to node {recipient_id} at {url}")

                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        app.logger.debug(f"Message sent successfully to node {recipient_id}")
                        return True
                    else:
                        app.logger.error(f"Failed to send message to node {recipient_id}, status: {response.status}")
                        return False
        except Exception as e:
            app.logger.error(f"Exception in send_message to node {recipient_id}: {e}")
            return False

    async def receive_message(self, message):
        app.logger.debug(f"Node {self.node_id} received: {message}")
        print(f"Node {self.node_id} received: {message}")
        message_type = message['type']
        data = message['data']

        if message_type == 'Pre-prepare':
            if not self.is_leader:
                for node_id in self.nodes:
                    if node_id != self.node_id:
                        await self.send_message(node_id, 'Prepare', {
                            'sequence_number': data['sequence_number'],
                            'digest': data['digest']
                        })
            self.pre_prepare(message['sender_id'], data['sequence_number'], data['digest'], data['request'])
        elif message_type == 'Prepare':
            app.logger.debug(
                f"Node {self.node_id} is receiving prepare for sequence {data['sequence_number']} with digest {data['digest']}")
            self.prepare(message['sender_id'], data['sequence_number'], data['digest'])

            if data['sequence_number'] in self.prepared and data['digest'] in self.prepared[
                data['sequence_number']] and len(self.prepared[data['sequence_number']][data['digest']]) >= 1:
                app.logger.debug(
                    f"Node {self.node_id} has enough prepares for sequence {data['sequence_number']} with digest {data['digest']}")
                for node_id in self.nodes:
                    if node_id != self.node_id:
                        await self.send_message(node_id, 'Commit', {
                            'sequence_number': data['sequence_number'],
                            'digest': data['digest']
                        })
                await self.commit(self.node_id, data['sequence_number'], data['digest'])

    def pre_prepare(self, sender_id, sequence_number, digest, request):
        if self.is_leader:
            return

        self.requests[sequence_number] = request

        if sequence_number not in self.prepared:
            self.prepared[sequence_number] = {}
        if digest not in self.prepared[sequence_number]:
            self.prepared[sequence_number][digest] = set()
        self.prepared[sequence_number][digest].add(self.node_id)

    def prepare(self, sender_id, sequence_number, digest):
        app.logger.debug(f"Node {self.node_id} is preparing sequence {sequence_number} with digest {digest}. sender_id = {sender_id}, self.prepared = {self.prepared}")

        if sequence_number not in self.prepared:
            self.prepared[sequence_number] = {}
        if digest not in self.prepared[sequence_number]:
            self.prepared[sequence_number][digest] = set()

        self.prepared[sequence_number][digest].add(self.node_id)

        app.logger.debug(
            f"Node {self.node_id} received Prepare from {sender_id} for sequence {sequence_number}, digest {digest}. Prepared: {self.prepared}")

    async def commit(self, sender_id, sequence_number, digest):
        if sequence_number not in self.committed:
            self.committed[sequence_number] = {}
        if digest not in self.committed[sequence_number]:
            self.committed[sequence_number][digest] = set()

        self.committed[sequence_number][digest].add(self.node_id)

        app.logger.debug(
            f"Node {self.node_id} received Commit from {sender_id} for sequence {sequence_number}, digest {digest}. Committed: {self.committed}")

        already_applied = hasattr(self, 'applied_transactions') and sequence_number in self.applied_transactions

        if not already_applied and sequence_number in self.committed and digest in self.committed[
            sequence_number] and len(
            self.committed[sequence_number][digest]) >= 1:
            if not hasattr(self, 'applied_transactions'):
                self.applied_transactions = set()
            self.applied_transactions.add(sequence_number)

            await self.apply_transaction(sequence_number, digest)

    # Новый метод для оповещения узлов о новом блоке
    async def broadcast_new_block(self, block_data):
        """Оповестить все узлы о новом блоке"""
        app.logger.debug(f"Broadcasting new block with hash {block_data['hash']}")

        for node_id, node in self.nodes.items():
            if node_id != self.node_id:  # Не отправляем блок самому себе
                try:
                    app.logger.debug(f"Sending new block to node {node_id}")
                    host = node.host
                    port = node.port

                    # Проверка доступности узла перед отправкой (опционально)
                    is_node_available = await self.check_node_availability(host, port)
                    if not is_node_available:
                        app.logger.warning(f"Node {node_id} не доступен, пропускаем отправку блока")
                        continue

                    async with aiohttp.ClientSession() as session:
                        url = f"http://{host}:{port}/receive_block"
                        payload = {
                            'sender_id': self.node_id,
                            'block': block_data
                        }
                        timeout = aiohttp.ClientTimeout(total=5)  # 5-секундный таймаут
                        async with session.post(url, json=payload, timeout=timeout) as response:
                            if response.status == 200:
                                app.logger.debug(f"Block successfully sent to node {node_id}")
                            else:
                                app.logger.error(f"Failed to send block to node {node_id}, status: {response.status}")
                except aiohttp.ClientConnectorError:
                    app.logger.warning(f"Node {node_id} недоступен (connection error)")
                except asyncio.TimeoutError:
                    app.logger.warning(f"Timeout при отправке блока узлу {node_id}")
                except Exception as e:
                    app.logger.error(f"Error sending block to node {node_id}: {e}")

    # Проверка доступности узла
    async def check_node_availability(self, host, port):
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{host}:{port}/health"  # Endpoint для проверки доступности
                timeout = aiohttp.ClientTimeout(total=2)  # 2-секундный таймаут
                async with session.get(url, timeout=timeout) as response:
                    return response.status == 200
        except:
            return False

    async def handle_request(self, sender_id, request_data):
        try:
            app.logger.debug(f"Node {self.node_id} handling request from client: {sender_id}")
            app.logger.debug(f"Request data: {request_data}")

            self.sequence_number += 1
            sequence_number = self.sequence_number

            request_data['timestamp'] = datetime.datetime.now().isoformat()

            request_string = json.dumps(request_data, sort_keys=True)
            request_digest = self.generate_digest(request_string.encode('utf-8'))

            self.requests[sequence_number] = request_string
            app.logger.debug(f"Created request with sequence_number {sequence_number}")

            # Флаг для отслеживания успешных отправок
            successful_sends = 0

            # Попытка отправить сообщения другим узлам
            for node_id in self.nodes:
                try:
                    app.logger.debug(f"Sending pre-prepare message to node {node_id}")
                    await self.send_message(node_id, 'Pre-prepare', {
                        'sequence_number': sequence_number,
                        'digest': request_digest,
                        'request': request_string
                    })
                    successful_sends += 1
                except Exception as e:
                    app.logger.error(f"Failed to send message to node {node_id}: {e}")

            # Вызываем apply_transaction напрямую в любом случае
            app.logger.debug(f"Applying transaction directly on node {self.node_id}")
            success, message = await self.apply_transaction(sequence_number, request_digest)

            if not success:
                app.logger.error(f"Failed to apply transaction: {message}")
                return False, message

            if not self.is_leader:
                app.logger.debug(f"Node {self.node_id} is not leader, calling pre_prepare locally")
                self.pre_prepare(self.node_id, sequence_number, request_digest, request_string)

            return True, "Transaction applied successfully."
        except Exception as e:
            app.logger.error(f"Error in handle_request: {e}")
            return False, str(e)

    async def apply_transaction(self, sequence_number, digest):
        app.logger.debug(f"Applying transaction {sequence_number} with digest {digest}")

        request = self.requests.get(sequence_number)
        if not request:
            app.logger.error(f"Request with sequence number {sequence_number} not found.")
            return False, "Request with sequence number not found."

        try:
            transaction_data = json.loads(request)
            app.logger.debug(f"Transaction data to apply: {transaction_data}")

            transaction_string = json.dumps(transaction_data, sort_keys=True).encode('utf-8')
            transaction_hash = hashlib.sha256(transaction_string).hexdigest()
            app.logger.debug(f"Generated transaction hash: {transaction_hash}")

            with app.app_context():
                try:
                    # Начинаем транзакцию
                    db.session.begin_nested()  # Создаем savepoint

                    # Проверка наличия товара для расходных документов
                    if transaction_data['ДокументID'] in [1, 3]:  # ID документов типа "Расходная накладная" и др.
                        if transaction_data['СкладОтправительID'] != transaction_data['СкладПолучательID']:
                            # Предварительная проверка наличия товара на складе
                            запас = Запасы.query.filter_by(
                                СкладID=transaction_data['СкладОтправительID'],
                                ТоварID=transaction_data['ТоварID']
                            ).first()

                            if not запас or запас.Количество < transaction_data['Количество']:
                                товар = Товары.query.get(transaction_data['ТоварID'])
                                склад = Склады.query.get(transaction_data['СкладОтправительID'])

                                кол_на_складе = запас.Количество if запас else 0
                                error_msg = f"Недостаточно товара '{товар.Наименование}' на складе '{склад.Название}'. " \
                                            f"Требуется: {transaction_data['Количество']}, в наличии: {кол_на_складе}"
                                db.session.rollback()
                                return False, error_msg

                    # Создаем запись о приходе/расходе
                    new_record = ПриходРасход(
                        СкладОтправительID=transaction_data['СкладОтправительID'],
                        СкладПолучательID=transaction_data['СкладПолучательID'],
                        ДокументID=transaction_data['ДокументID'],
                        ТоварID=transaction_data['ТоварID'],
                        Количество=transaction_data['Количество'],
                        Единица_ИзмеренияID=transaction_data['Единица_ИзмеренияID'],
                        TransactionHash=transaction_hash,
                        Timestamp=transaction_data.get('timestamp')
                    )

                    db.session.add(new_record)
                    db.session.flush()  # Чтобы получить ID новой записи

                    # Обновляем запасы
                    # Обработка прихода на склад-получатель
                    success, message = update_запасы(
                        transaction_data['СкладПолучательID'],
                        transaction_data['ТоварID'],
                        transaction_data['Количество']
                    )

                    if not success:
                        db.session.rollback()
                        return False, message

                    # Обработка расхода со склада-отправителя, если это разные склады
                    if transaction_data['СкладОтправительID'] != transaction_data['СкладПолучательID']:
                        success, message = update_запасы(
                            transaction_data['СкладОтправительID'],
                            transaction_data['ТоварID'],
                            -transaction_data['Количество']
                        )

                        if not success:
                            db.session.rollback()
                            return False, message

                    # Создаем новый блок для блокчейна
                    last_block = BlockchainBlock.query.order_by(BlockchainBlock.index.desc()).first()

                    if last_block:
                        next_index = last_block.index + 1
                        previous_hash = last_block.hash
                    else:
                        # Создаем генезис-блок, если блоков еще нет
                        next_index = 0
                        previous_hash = '0' * 64

                    # Создаем новый блок
                    new_block = Block(
                        index=next_index,
                        timestamp=datetime.datetime.now(),
                        transactions=[transaction_data],
                        previous_hash=previous_hash
                    )

                    # Сохраняем блок в БД
                    block_db = BlockchainBlock(
                        index=new_block.index,
                        timestamp=new_block.timestamp,
                        transactions=json.dumps(new_block.transactions, ensure_ascii=False),
                        previous_hash=new_block.previous_hash,
                        hash=new_block.hash,
                        node_id=self.node_id
                    )

                    db.session.add(block_db)
                    db.session.commit()  # Финальный коммит всех изменений

                    app.logger.info(f"Transaction {transaction_hash} successfully saved to DB")
                    app.logger.info(f"Added new block to chain: {new_block.hash}")

                    # Оповещаем другие узлы о новом блоке
                    await self.broadcast_new_block(new_block.to_dict())
                    return True, "Transaction applied successfully."

                except Exception as db_error:
                    db.session.rollback()
                    app.logger.error(f"Database error while saving transaction: {db_error}")
                    return False, f"Database error while saving transaction: {db_error}"

        except json.JSONDecodeError as json_error:
            app.logger.error(f"Failed to decode JSON request: {json_error}")
            return False, "Failed to decode JSON request."
        except Exception as e:
            app.logger.error(f"Unexpected error in apply_transaction: {e}")
            return False, f"Unexpected error in apply_transaction: {e}"

    def generate_digest(self, message):
        return hashlib.sha256(message).hexdigest()

    def calculate_hash(self):
        data = str(self.index) + str(self.timestamp) + json.dumps(self.transactions, sort_keys=True) + str(self.previous_hash)
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

# Создаем узлы блокчейна
nodes = {
    0: Node(0, {1: {'host': 'localhost', 'port': 5001}, 2: {'host': 'localhost', 'port': 5002}}, 'localhost', 5000),
    1: Node(1, {0: {'host': 'localhost', 'port': 5000}, 2: {'host': 'localhost', 'port': 5002}}, 'localhost', 5001),
    2: Node(2, {0: {'host': 'localhost', 'port': 5000}, 1: {'host': 'localhost', 'port': 5001}}, 'localhost', 5002)
}

for node_id, node in nodes.items():
    node.nodes = {k: v for k, v in nodes.items() if k != node_id}

with app.app_context():
    csrf = CSRFProtect(app)

def serialize_data(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


@app.route('/receive_block', methods=['POST'])
async def receive_block():
    try:
        data = await request.get_json()
        sender_id = data.get('sender_id')
        block_data = data.get('block')

        app.logger.debug(f"Received block from node {sender_id}: {block_data['hash']}")

        # Проверка валидности блока
        if not block_data or not all(
                k in block_data for k in ['index', 'timestamp', 'transactions', 'previous_hash', 'hash']):
            app.logger.error("Invalid block data received")
            return jsonify({"status": "error", "message": "Invalid block data"}), 400

        # Проверка, существует ли уже блок с таким хешем
        with app.app_context():
            existing_block = BlockchainBlock.query.filter_by(hash=block_data['hash']).first()
            if existing_block:
                app.logger.debug(f"Block with hash {block_data['hash']} already exists")
                return jsonify({"status": "ok", "message": "Block already exists"}), 200

            # Проверка, что предыдущий блок существует
            if block_data['index'] > 0:
                prev_block = BlockchainBlock.query.filter_by(hash=block_data['previous_hash']).first()
                if not prev_block:
                    app.logger.error(f"Previous block with hash {block_data['previous_hash']} not found")
                    return jsonify({"status": "error", "message": "Previous block not found"}), 400

            # Сохраняем блок в БД
            try:
                new_block_db = BlockchainBlock(
                    index=block_data['index'],
                    timestamp=datetime.datetime.fromisoformat(block_data['timestamp']),
                    transactions=json.dumps(block_data['transactions'], ensure_ascii=False),
                    # Для корректного хранения русских букв
                    previous_hash=block_data['previous_hash'],
                    hash=block_data['hash'],
                    node_id=sender_id
                )

                db.session.add(new_block_db)
                db.session.commit()
                app.logger.info(f"Added received block with hash {block_data['hash']} to blockchain")
            except Exception as db_error:
                db.session.rollback()
                app.logger.error(f"Error saving received block to database: {db_error}")
                return jsonify({"status": "error", "message": str(db_error)}), 500

        return jsonify({"status": "ok", "message": "Block received and saved"}), 200
    except Exception as e:
        app.logger.error(f"Error processing received block: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


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

        # История движения товаров (последние 10 транзакций)
        latest_transactions = db.session.query(
            ПриходРасход, Склады.Название.label('sender_warehouse'),
            Склады.Название.label('receiver_warehouse'),
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
            Склады, ПриходРасход.СкладОтправительID == Склады.СкладID
        ).join(
            Склады, ПриходРасход.СкладПолучательID == Склады.СкладID
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

        return render_template(
            'inventory_report.html',
            inventory_data=inventory_data,
            total_by_product=total_by_product,
            total_by_warehouse=total_by_warehouse,
            latest_transactions=latest_transactions,
            zero_inventory=zero_inventory
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
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Получаем текущий узел
        node_id = 0
        if current_user.role == 'admin':
            node_id = 0
        elif current_user.role == 'north':
            node_id = 1
        elif current_user.role == 'south':
            node_id = 2

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


@app.route('/blockchain')
@login_required
def view_blockchain():
    app.logger.debug("Entering view_blockchain function")

    try:
        # Получаем все блоки из БД, сортируем по индексу
        blocks = BlockchainBlock.query.order_by(BlockchainBlock.index).all()

        # Получаем справочные данные из базы с корректными именами моделей
        warehouses = {w.СкладID: w.Название for w in Склады.query.all()}
        documents = {d.ДокументID: d.Тип_документа for d in Тип_документа.query.all()}
        products = {p.ТоварID: p.Наименование for p in Товары.query.all()}
        units = {u.Единица_ИзмеренияID: u.Единица_Измерения for u in Единица_измерения.query.all()}

        # Преобразуем блоки для отображения
        formatted_blocks = []
        for block in blocks:
            try:
                # Загружаем транзакции из JSON с поддержкой русских символов
                transactions_json = block.transactions
                transactions = json.loads(transactions_json)

                # Обогащаем данные транзакций названиями вместо ID
                for transaction in transactions:
                    if isinstance(transaction, dict):  # Проверяем, что это словарь
                        # Только если это не генезис-блок с сообщением
                        if 'message' not in transaction:
                            # Добавляем названия для ID, если они существуют
                            transaction['sender_name'] = warehouses.get(transaction.get('СкладОтправительID'),
                                                                        'Неизвестно')
                            transaction['receiver_name'] = warehouses.get(transaction.get('СкладПолучательID'),
                                                                          'Неизвестно')
                            transaction['document_name'] = documents.get(transaction.get('ДокументID'), 'Неизвестно')
                            transaction['product_name'] = products.get(transaction.get('ТоварID'), 'Неизвестно')
                            transaction['unit_name'] = units.get(transaction.get('Единица_ИзмеренияID'), 'Неизвестно')

                # Формируем блок с обогащенными данными
                formatted_blocks.append({
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'transactions': transactions,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'node_id': block.node_id,
                    'is_genesis': block.index == 0  # Признак генезис-блока
                })
            except json.JSONDecodeError as e:
                app.logger.error(f"Error decoding transactions for block {block.id}: {e}")

        app.logger.debug("Exiting view_blockchain function")
        return render_template('blockchain.html', blocks=formatted_blocks)
    except Exception as e:
        app.logger.error(f"Error in view_blockchain: {e}")
        flash(f'Ошибка при загрузке блокчейна: {e}', 'danger')
        return redirect(url_for('index'))

@app.route('/receive_message', methods=['POST'])
async def receive_message():
    app.logger.debug('Entering receive_message function')
    message = request.json
    node_id = message['sender_id']
    node = nodes.get(node_id)
    if node:
        await node.receive_message(message)
        app.logger.debug('Exiting receive_message function')
        return jsonify({'success': True})
    else:
        app.logger.debug('Exiting receive_message function with error')
        return jsonify({'success': False, 'message': 'Node not found'}), 404

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
        invitation.used_at = datetime.datetime.utcnow()
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

@app.route('/', methods=['GET', 'POST'])
@login_required
async def index():
    app.logger.debug('Entering index function')
    connection_status = ""

    try:
        db.session.execute(text("SELECT 1"))
        connection_status = "Соединение с базой данных установлено!"
        app.logger.info(connection_status)
    except Exception as e:
        connection_status = f"Ошибка подключения к базе данных: {e}"
        app.logger.error(connection_status)
        flash(connection_status, 'danger')

    form = PrihodRashodForm()
    transaction_data = None

    # Получаем записи ПриходРасход и текущие запасы
    try:
        prihod_rashod_records = ПриходРасход.query.order_by(ПриходРасход.ПриходРасходID.desc()).all()
        current_inventory = Запасы.query.all()
        app.logger.debug(f"Загружено {len(prihod_rashod_records)} записей из базы данных")
    except Exception as db_error:
        error_msg = f"Ошибка при загрузке записей из базы данных: {db_error}"
        app.logger.error(error_msg)
        flash(error_msg, 'danger')
        prihod_rashod_records = []
        current_inventory = []

    if request.method == 'POST':
        app.logger.debug(f"POST-запрос получен: {request.form}")

        # Проверяем, является ли запрос AJAX
        is_ajax_request = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        if not form.validate():
            app.logger.error(f"Ошибки валидации формы: {form.errors}")
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
            app.logger.debug("Форма успешно прошла валидацию")

            try:
                transaction_data = {
                    'СкладОтправительID': form.СкладОтправительID.data,
                    'СкладПолучательID': form.СкладПолучательID.data,
                    'ДокументID': form.ДокументID.data,
                    'ТоварID': form.ТоварID.data,
                    'Количество': form.Количество.data,
                    'Единица_ИзмеренияID': form.Единица_ИзмеренияID.data,
                    'timestamp': datetime.datetime.now().isoformat()
                }

                app.logger.debug(f"Сформированы данные транзакции: {transaction_data}")

                node_id = None
                if current_user.role == 'admin':
                    node_id = 0
                elif current_user.role == 'north':
                    node_id = 1
                elif current_user.role == 'south':
                    node_id = 2
                else:
                    error_msg = f"Неподдерживаемая роль пользователя: {current_user.role}"
                    app.logger.error(error_msg)
                    if is_ajax_request:
                        return jsonify({'success': False, 'message': error_msg}), 400
                    else:
                        flash(error_msg, 'danger')
                        return redirect(url_for('index'))

                app.logger.debug(f"Выбран узел {node_id} для обработки транзакции")

                if node_id not in nodes:
                    error_msg = f"Узел с ID {node_id} не найден"
                    app.logger.error(error_msg)
                    if is_ajax_request:
                        return jsonify({'success': False, 'message': error_msg}), 400
                    else:
                        flash(error_msg, 'danger')
                        return redirect(url_for('index'))

                try:
                    # Функция handle_request должна возвращать кортеж (success, message)
                    success, message = await nodes[node_id].handle_request(current_user.id, transaction_data)
                    if success:
                        app.logger.debug(f"Транзакция успешно отправлена на узел {node_id}")
                        if is_ajax_request:
                            # Для AJAX-запроса возвращаем JSON с успехом
                            return jsonify({'success': True, 'message': 'Запись успешно добавлена'}), 200
                        else:
                            flash('Запись успешно добавлена', 'success')
                    else:
                        app.logger.error(f"Ошибка при обработке запроса узлом {node_id}: {message}")
                        if is_ajax_request:
                            return jsonify(
                                {'success': False, 'message': f"Ошибка при добавлении записи: {message}"}), 400
                        else:
                            flash(f"Ошибка при добавлении записи: {message}", 'danger')
                except Exception as node_error:
                    error_msg = f"Ошибка при обработке запроса узлом {node_id}: {node_error}"
                    app.logger.error(error_msg)
                    if is_ajax_request:
                        return jsonify({'success': False, 'message': error_msg}), 500
                    else:
                        flash(error_msg, 'danger')

            except Exception as e:
                error_msg = f"Ошибка при подготовке транзакции: {e}"
                app.logger.error(error_msg)
                if is_ajax_request:
                    return jsonify({'success': False, 'message': error_msg}), 500
                else:
                    flash(error_msg, 'danger')

            if not is_ajax_request:
                return redirect(url_for('index'))

    app.logger.debug('Exiting index function')
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
    app.logger.debug('Entering check_data_integrity function')
    try:
        record = ПриходРасход.query.get(record_id)
        if not record:
            app.logger.debug('Exiting check_data_integrity function with error')
            return {
                'success': False,
                'message': "Record not found",
                'details': "Запись с указанным ID не найдена в базе данных"
            }

        transaction_data = {
            'СкладОтправительID': record.СкладОтправительID,
            'СкладПолучательID': record.СкладПолучательID,
            'ДокументID': record.ДокументID,
            'ТоварID': record.ТоварID,
            'Количество': record.Количество,
            'Единица_ИзмеренияID': record.Единица_ИзмеренияID,
            'timestamp': record.Timestamp
        }

        transaction_data_string = json.dumps(transaction_data, sort_keys=True)
        computed_hash = hashlib.sha256(transaction_data_string.encode('utf-8')).hexdigest()

        if computed_hash == record.TransactionHash:
            app.logger.debug('Exiting check_data_integrity function')
            return {
                'success': True,
                'message': "Целостность данных подтверждена",
                'details': "Хэш транзакции в блокчейне совпадает с вычисленным хэшом данных",
                'transaction_data': transaction_data,
                'computed_hash': computed_hash,
                'stored_hash': record.TransactionHash
            }
        else:
            app.logger.debug('Exiting check_data_integrity function with error')
            return {
                'success': False,
                'message': "Обнаружено несоответствие данных",
                'details': f"Вычисленный хэш ({computed_hash}) не совпадает с хэшем в записи ({record.TransactionHash})",
                'transaction_data': transaction_data,
                'computed_hash': computed_hash,
                'stored_hash': record.TransactionHash
            }
    except Exception as e:
        app.logger.debug('Exiting check_data_integrity function with error')
        return {
            'success': False,
            'message': "Ошибка при проверке целостности данных",
            'details': str(e)
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
                    return False, f"Невозможно отгрузить {abs(количество)} единиц товара '{товар.Наименование}' со склада '{склад.Название}', так как товар отсутствует на складе"

                if запас.Количество + количество < 0:
                    return False, f"Недостаточно товара '{товар.Наименование}' на складе '{склад.Название}'. Требуется: {abs(количество)}, в наличии: {запас.Количество}"

            запас = Запасы.query.filter_by(СкладID=склад_id, ТоварID=товар_id).first()
            if запас:
                # Обновляем существующую запись
                запас.Количество += количество
                запас.Дата_обновления = datetime.datetime.now().date()
                db.session.commit()

                операция = "приход" if количество > 0 else "расход"
                return True, f"Обновлены запасы товара '{товар.Наименование}' на складе '{склад.Название}'. {операция}: {abs(количество)}, новый остаток: {запас.Количество}"
            else:
                # Создаем новую запись
                # Для расхода уже проверено выше, что такой ситуации не может быть
                единица_измерения = товар.Единица_ИзмеренияID

                new_запас = Запасы(
                    СкладID=склад_id,
                    ТоварID=товар_id,
                    Количество=количество,
                    Дата_обновления=datetime.datetime.now().date(),
                    Единица_ИзмеренияID=единица_измерения  # Используем единицу измерения из товара
                )
                db.session.add(new_запас)
                db.session.commit()

                return True, f"Создана запись о товаре '{товар.Наименование}' на складе '{склад.Название}'. Начальный остаток: {количество}"

        except Exception as e:
            db.session.rollback()
            error_message = f"Ошибка при обновлении запасов: {e}"
            app.logger.error(error_message)
            return False, error_message

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

        return jsonify({
            'success': True,
            'item_id': item.ТоварID,
            'item_name': item.Наименование,
            'description': item.Описание if hasattr(item, 'Описание') else 'Не указано',
            'unit': unit.Единица_Измерения if unit else 'не указано',
            'unit_id': item.Единица_ИзмеренияID
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    import sys

    # Параметры узла по умолчанию
    node_id = 0
    port = 5000

    # Аргументы командной строки (для локального тестирования)
    if len(sys.argv) > 1:
        try:
            node_id = int(sys.argv[1])
            if node_id in nodes:
                port = nodes[node_id].port
            else:
                print(f"Ошибка: Узел с ID {node_id} не существует")
                sys.exit(1)
        except ValueError:
            print("Ошибка: ID узла должен быть целым числом")
            sys.exit(1)

    # Для Railway - используем порт из переменной окружения
    port = int(os.environ.get('PORT', port))

    print(f"Запуск узла {node_id} на порту {port}")
    app.run(host='0.0.0.0', port=port)