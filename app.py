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

load_dotenv()

app = Flask(__name__)
CORS(app)

NODE_DOMAINS = {
    0: os.environ.get('NODE0_DOMAIN', 'blockchaininventory0.up.railway.app'),
    1: os.environ.get('NODE1_DOMAIN', 'blockchaininventory1.up.railway.app'),
    2: os.environ.get('NODE2_DOMAIN', 'blockchaininventory2.up.railway.app'),
    3: os.environ.get('NODE3_DOMAIN', 'blockchaininventory3.up.railway.app')
}

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
        with app.app_context():
            # Проверяем, существует ли уже генезис-блок в БД
            existing_genesis = BlockchainBlock.query.filter_by(index=0).first()
            if existing_genesis:
                return Block(
                    index=0,
                    timestamp=existing_genesis.timestamp,
                    transactions=json.loads(existing_genesis.transactions),
                    previous_hash=existing_genesis.previous_hash
                )
        
        # Создаем новый генезис-блок только если его нет
        genesis = Block(0, datetime.now(timezone.utc), [], "0")
        
        # Сохраняем в БД
        with app.app_context():
            genesis_db = BlockchainBlock(
                index=genesis.index,
                timestamp=genesis.timestamp,
                transactions=json.dumps(genesis.transactions, ensure_ascii=False),
                previous_hash=genesis.previous_hash,
                hash=genesis.hash,
                node_id=self.node_id,
                confirming_node_id=self.node_id,
                confirmed=True  # Генезис-блок всегда считается подтвержденным
            )
            db.session.add(genesis_db)
            db.session.commit()
        
        return genesis

    def get_last_block(self):
        return self.chain[-1]

    # В начале работы узла
    async def sync_blockchain(self):
        app.logger.info(f"Node {self.node_id} started sync...")
        app.logger.info(f"Node {self.node_id} starts blockchain sync")
        
        try:
            # Получаем последний локальный блок
            with app.app_context():
                local_last_block = BlockchainBlock.query.order_by(BlockchainBlock.index.desc()).first()
                local_index = local_last_block.index if local_last_block else -1
            
            # Запрашиваем высоту блокчейна у других узлов
            heights = {}
            for node_id, node in self.nodes.items():
                if node_id != self.node_id:
                    try:
                        async with aiohttp.ClientSession() as session:
                            url = f"https://{node.host}/get_blockchain_height"
                            async with session.get(url, timeout=5) as response:
                                if response.status == 200:
                                    data = await response.json()
                                    heights[node_id] = data.get('height', -1)
                    except Exception as e:
                        app.logger.error(f"Error getting height from node {node_id}: {e}")
                        continue
            
            # Если не получили ответы от других узлов, выходим
            if not heights:
                app.logger.warning("No responses from other nodes during sync")
                return
            
            # Находим максимальную высоту
            max_height = max(heights.values())
            
            # Если наш блокчейн короче, синхронизируем недостающие блоки
            if max_height > local_index:
                app.logger.info(f"Need to sync from {local_index + 1} to {max_height}")
                for i in range(local_index + 1, max_height + 1):
                    # Пытаемся получить блок с каждого узла
                    for node_id in heights:
                        if heights[node_id] >= i:
                            success = await self.request_block_from_node(node_id, i)
                            if success:
                                break
        except Exception as e:
            app.logger.error(f"Error during blockchain sync: {e}")

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

    async def send_message(self, recipient_id, message_type, data):
        try:
            recipient = self.nodes.get(recipient_id)
            if not recipient:
                app.logger.error(f"Recipient node {recipient_id} not found in nodes list")
                return False
            host = recipient.host
            port = recipient.port
            async with aiohttp.ClientSession() as session:
                url = f"https://{host}:{port}/node_message"
                payload = {
                    'sender_id': self.node_id,
                    'message_type': message_type,
                    'data': data
                }
                app.logger.debug(f"Sending {message_type} to node {recipient_id} at {url}")

                async with session.post(url, json=payload, headers={'Content-Type': 'application/json'}) as response:
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
        if self.is_leader:  # Только лидер инициирует Pre-prepare
            return

        # Сохраняем запрос
        self.requests[sequence_number] = request

        if sequence_number not in self.prepared:
            self.prepared[sequence_number] = {}
        if digest not in self.prepared[sequence_number]:
            self.prepared[sequence_number][digest] = set()

        self.prepared[sequence_number][digest].add(sender_id)

        # Если это не лидер, рассылаем Prepare
        if not self.is_leader:
            for node_id in self.nodes:
                if node_id != self.node_id:
                    asyncio.create_task(self.send_message(node_id, 'Prepare', {
                        'sequence_number': sequence_number,
                        'digest': digest
                    }))

    async def prepare(self, sender_id, sequence_number, digest):
        if sequence_number not in self.prepared:
            self.prepared[sequence_number] = {}
        if digest not in self.prepared[sequence_number]:
            self.prepared[sequence_number][digest] = set()

        self.prepared[sequence_number][digest].add(sender_id)

        # Проверяем, что собрано 2f + 1 подтверждений (f = 1 при N=4)
        if len(self.prepared[sequence_number][digest]) >= 2 * 1 + 1:  # 2f + 1 = 3
            # Рассылаем Commit всем узлам
            for node_id in self.nodes:
                if node_id != self.node_id:
                    await self.send_message(node_id, 'Commit', {
                        'sequence_number': sequence_number,
                        'digest': digest
                    })
            # Локально подтверждаем Commit
            await self.commit(self.node_id, sequence_number, digest)

    async def commit(self, sender_id, sequence_number, digest):
        if sequence_number not in self.committed:
            self.committed[sequence_number] = {}
        if digest not in self.committed[sequence_number]:
            self.committed[sequence_number][digest] = set()

        self.committed[sequence_number][digest].add(sender_id)

        # Проверяем кворум (2f + 1 подтверждений)
        if len(self.committed[sequence_number][digest]) >= 2 * 1 + 1:  # 2f + 1 = 3
            # Применяем транзакцию, только если она еще не была применена
            if not hasattr(self, 'applied_transactions'):
                self.applied_transactions = set()
            if sequence_number not in self.applied_transactions:
                self.applied_transactions.add(sequence_number)
                await self.apply_transaction(sequence_number, digest)

    async def broadcast_new_block(self, block_data):
        """Рассылает новый блок всем узлам сети и сохраняет подтверждения в БД"""
        # 1. Подготовка и валидация блока (оставляем ваш текущий код)
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
        if isinstance(block_data, Block):
            block_data = block_data.to_dict()
    
        required_block_fields = ['index', 'transactions', 'previous_hash', 'hash']
        if not all(field in block_data for field in required_block_fields):
            app.logger.error(f"Invalid block structure. Missing fields: {required_block_fields}")
            return 0, len(self.nodes)
    
        try:
            if 'timestamp' not in block_data:
                block_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            else:
                dt = datetime.fromisoformat(block_data['timestamp'].replace('Z', '+00:00'))
                block_data['timestamp'] = dt.isoformat()
        except ValueError as e:
            app.logger.error(f"Invalid timestamp format: {str(e)}")
            return 0, len(self.nodes)
    
        # 2. Подготовка payload (упрощаем)
        payload = {
            'sender_id': self.node_id,
            'block': block_data
        }
    
        # 3. Сохраняем блок от создателя (ваш текущий код)
        with app.app_context():
            try:
                existing_block = BlockchainBlock.query.filter_by(
                    index=block_data['index'],
                    node_id=self.node_id,
                    confirming_node_id=self.node_id
                ).first()
                
                if not existing_block:
                    new_block = BlockchainBlock(
                        index=block_data['index'],
                        timestamp=datetime.fromisoformat(block_data['timestamp']),
                        transactions=json.dumps(block_data['transactions']),
                        previous_hash=block_data['previous_hash'],
                        hash=block_data['hash'],
                        node_id=self.node_id,
                        confirming_node_id=self.node_id,
                        confirmed=False
                    )
                    db.session.add(new_block)
                db.session.commit()
            except Exception as e:
                app.logger.error(f"Error saving block to DB: {e}")
                db.session.rollback()
                return 0, len(self.nodes)
    
        # 4. Рассылка блока другим узлам (исправленная версия)
        confirmations = 1  # Текущий узел всегда подтверждает
        confirming_nodes = {self.node_id}
        timeout = aiohttp.ClientTimeout(total=10)
    
        for node_id, node in self.nodes.items():
            if node_id == self.node_id:
                continue  # Не отправляем себе
    
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    url = f"https://{node.host}/receive_block"
                    async with session.post(url, json=payload, timeout=timeout) as response:
                        if response.status == 200:
                            confirmations += 1
                            confirming_nodes.add(node_id)
                            
                            # Сохраняем подтверждение от этого узла
                            with app.app_context():
                                confirmation = BlockchainBlock(
                                    index=block_data['index'],
                                    timestamp=datetime.fromisoformat(block_data['timestamp']),
                                    transactions=json.dumps(block_data['transactions']),
                                    previous_hash=block_data['previous_hash'],
                                    hash=block_data['hash'],
                                    node_id=self.node_id,  # Создатель блока
                                    confirming_node_id=node_id,  # Кто подтвердил
                                    confirmed=False
                                )
                                db.session.add(confirmation)
                                db.session.commit()
            except Exception as e:
                app.logger.error(f"Error broadcasting to node {node_id}: {e}")
    
        # 5. Проверяем консенсус (ваш текущий код)
        required_confirmations = (len(self.nodes) // 3 * 2) + 1
        consensus_reached = confirmations >= required_confirmations
    
        if consensus_reached:
            with app.app_context():
                BlockchainBlock.query.filter_by(index=block_data['index']).update({'confirmed': True})
                db.session.commit()
            app.logger.info(f"Consensus reached for block #{block_data['index']} ({confirmations}/{len(self.nodes)})")
        else:
            app.logger.warning(f"Consensus not reached for block #{block_data['index']} ({confirmations}/{len(self.nodes)})")
            await self.sync_blockchain()
    
        return confirmations, len(self.nodes)

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


    async def handle_request(self, sender_id, request_data):
        try:
            app.logger.debug(f"Node {self.node_id} handling request from client: {sender_id}")
            app.logger.debug(f"Request data: {request_data}")
            self.sequence_number += 1
            sequence_number = self.sequence_number

            # Добавляем user_id и timestamp в данные запроса
            request_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            request_data['user_id'] = sender_id
            request_string = json.dumps(request_data, sort_keys=True)
            request_digest = self.generate_digest(request_string.encode('utf-8'))

            self.requests[sequence_number] = request_string
            app.logger.debug(f"Created request with sequence_number {sequence_number}")

            # Применяем транзакцию
            success, message = await self.apply_transaction(sequence_number, request_digest)

            if not success:
                app.logger.error(f"Failed to apply transaction: {message}")
                return False, message

            if not self.is_leader:
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
                        # Если timestamp строка - проверяем валидность
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

                    # Создание нового блока
                    last_block = BlockchainBlock.query.order_by(BlockchainBlock.index.desc()).first()
                    new_block = Block(
                        index=last_block.index + 1 if last_block else 0,
                        timestamp=datetime.now(timezone.utc),
                        transactions=[transaction_data],
                        previous_hash=last_block.hash if last_block else '0' * 64
                    )

                    # Сохранение в БД
                    block_db = BlockchainBlock(
                        index=new_block.index,
                        timestamp=new_block.timestamp,
                        transactions=json.dumps(new_block.transactions, ensure_ascii=False),
                        previous_hash=new_block.previous_hash,
                        hash=new_block.hash,
                        node_id=self.node_id,
                        confirming_node_id=self.node_id
                    )

                    db.session.add(new_record)
                    db.session.add(block_db)
                    db.session.commit()

                    app.logger.info(f"Transaction {transaction_hash} committed. Block #{new_block.index} added")

                    # Рассылка блока другим узлам
                    broadcast_success = await self.broadcast_new_block({
                        'index': new_block.index,
                        'timestamp': new_block.timestamp.isoformat(),
                        'transactions': new_block.transactions,
                        'previous_hash': new_block.previous_hash,
                        'hash': new_block.hash
                    })

                    if not broadcast_success:
                        app.logger.warning("Block broadcast failed, but transaction was committed")

                    return True, "Transaction applied successfully"

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

# Создаем узлы блокчейна
# Используем динамическое создание на основе переменных окружения:
NODE_ID = int(os.environ.get('NODE_ID', 0))
PORT = int(os.environ.get('PORT', 5000))
DOMAIN_PREFIX = os.environ.get('DOMAIN_PREFIX', '')


nodes = {}
for i in range(4):
    nodes[i] = Node(
        node_id=i,
        nodes={
            j: Node(  # Создаем объекты Node для других узлов
                node_id=j,
                nodes={},  # Пустой словарь, так как он будет заполнен позже
                host=NODE_DOMAINS[j],
                port=443  # Railway использует 443 для HTTPS
            ) for j in range(4) if j != i
        },
        host=NODE_DOMAINS[i],  # Используем домен из переменных окружения
        port=443  # Railway использует 443 для HTTPS
    )
current_node = nodes[NODE_ID]


for node_id, node in nodes.items():
    node.nodes = {k: v for k, v in nodes.items() if k != node_id}

with app.app_context():
    csrf = CSRFProtect(app)

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

@app.route('/receive_block', methods=['POST'])
async def receive_block():
    app.logger.debug(f"Received block from {sender_id}")
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    block_data = data["block"]
    sender_id = data["sender_id"]

    # Проверяем, что блок валиден
    last_block = BlockchainBlock.query.order_by(BlockchainBlock.index.desc()).first()
    if block_data["index"] != (last_block.index + 1 if last_block else 0):
        return jsonify({"error": "Invalid block index"}), 400

    # Сохраняем блок в БД как подтверждение
    new_block = BlockchainBlock(
        index=block_data["index"],
        timestamp=datetime.fromisoformat(block_data["timestamp"]),
        transactions=json.dumps(block_data["transactions"]),
        previous_hash=block_data["previous_hash"],
        hash=block_data["hash"],
        node_id=sender_id,  # Кто создал блок
        confirming_node_id=current_node.node_id,  # Мы подтверждаем
        confirmed=False
    )
    db.session.add(new_block)
    db.session.commit()

    return jsonify({"status": "Block accepted"}), 200


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
        for index in block_indices:
            index = index[0]
            
            # Получаем все подтверждения для этого блока
            confirmations = BlockchainBlock.query.filter_by(index=index).all()
            
            if not confirmations:
                continue

            # Группируем по node_id (создателям блоков)
            blocks_by_creator = {}
            for conf in confirmations:
                if conf.node_id not in blocks_by_creator:
                    blocks_by_creator[conf.node_id] = []
                blocks_by_creator[conf.node_id].append(conf)

            # Для каждого создателя блока создаем запись
            for creator_id, blocks in blocks_by_creator.items():
                main_block = blocks[0]
                
                try:
                    transactions = json.loads(main_block.transactions)
                    
                    # Получаем список уникальных узлов, подтвердивших этот блок
                    confirming_nodes = list({b.confirming_node_id for b in blocks})
                    
                    # Проверяем достижение консенсуса
                    required_confirmations = (len(nodes) // 3 * 2) + 1
                    consensus_reached = len(confirming_nodes) >= required_confirmations
                    
                    # Обновляем статус подтверждения в БД
                    if consensus_reached:
                        for b in blocks:
                            b.confirmed = True
                        db.session.commit()

                    formatted_blocks.append({
                        'index': index,
                        'timestamp': main_block.timestamp,
                        'transactions': transactions,
                        'previous_hash': main_block.previous_hash,
                        'hash': main_block.hash,
                        'node_id': creator_id,
                        'is_genesis': index == 0,
                        'confirmations': len(confirming_nodes),
                        'total_nodes': len(nodes),
                        'confirming_nodes': confirming_nodes,
                        'consensus_reached': consensus_reached
                    })
                except json.JSONDecodeError as e:
                    app.logger.error(f"Error decoding transactions for block {index}: {e}")
                    continue

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
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

                app.logger.debug(f"Сформированы данные транзакции: {transaction_data}")

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
    nodes_info = []
    for node_id, node in nodes.items():
        # Для текущего узла (node_id=0) всегда возвращаем True
        if node_id == 0:
            is_online = True
        else:
            try:
                # Используем асинхронную проверку с таймаутом
                async with aiohttp.ClientSession() as session:
                    url = f"https://{node.host}:{node.port}/health"
                    try:
                        async with session.get(url, timeout=5) as response:
                            is_online = response.status == 200
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        is_online = False
            except Exception as e:
                app.logger.error(f"Error checking node {node_id} status: {e}")
                is_online = False
        
        # Получаем количество блоков
        with app.app_context():
            block_count = BlockchainBlock.query.filter_by(node_id=node_id).count()

        nodes_info.append({
            'node_id': node_id,
            'host': node.host,
            'port': node.port,
            'is_online': is_online,
            'block_count': block_count,
            'is_leader': node_id == 0
        })

    return render_template('nodes_status.html', nodes=nodes_info)


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
@csrf.exempt
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

def start_sync(node):
    """Запуск синхронизации в отдельном потоке"""
    import threading
    def run_sync():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(node.sync_blockchain())
        loop.close()
    
    thread = threading.Thread(target=run_sync, daemon=True)
    thread.start()

if __name__ == '__main__':
    # Инициализация узла
    current_node = nodes[NODE_ID]
    
    # Запуск синхронизации в фоне
    start_sync(current_node)
    
    # Запуск Flask-приложения
    app.run(host='0.0.0.0', port=PORT)
