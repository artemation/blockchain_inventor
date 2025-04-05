# blockchain.py
import hashlib
import time
import json
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

class Block:
    def __init__(self, index, timestamp, data, previous_hash, sender, signature):  # Добавляем sender и signature
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.sender = sender  # Кто отправил транзакцию
        self.signature = signature  # Подпись транзакции
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data_string = f"{self.index}{self.timestamp}{self.data}{self.previous_hash}{self.sender}{self.signature}"  # Добавляем sender и signature
        return hashlib.sha256(data_string.encode('utf-8')).hexdigest()

    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'sender': self.sender,
            'signature': self.signature,
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data):
        # Используем get() с значениями по умолчанию, чтобы избежать KeyError
        sender = data.get('sender', "Unknown")  # Значение по умолчанию, если 'sender' нет
        signature = data.get('signature', "Unknown")  # Значение по умолчанию, если 'signature' нет
        return cls(data['index'], data['timestamp'], data['data'], data['previous_hash'], sender, signature)

    def __str__(self):  # Добавим для удобного просмотра
        return f"Block #{self.index} | Timestamp: {self.timestamp} | Data: {self.data} | Sender: {self.sender} | Signature: {self.signature} | Hash: {self.hash} | Previous Hash: {self.previous_hash}"

class Blockchain:
    def __init__(self, chain_file='blockchain.json'):
        print("Initializing Blockchain...")
        self.chain_file = chain_file
        print(f"Chain file: {self.chain_file}")
        self.chain = self.load_chain()
        print(f"Chain after load_chain: {self.chain}")

        if self.chain is None:
            print("Blockchain load failed, creating genesis block.")
            self.chain = [self.create_genesis_block()]
            print(f"Genesis block created: {self.chain}")
            self.save_chain()
        else:
            if not self.is_chain_valid():
                print("Blockchain loaded from file is invalid! Clearing the chain and creating a new genesis block.")
                self.chain = [self.create_genesis_block()]
                self.save_chain()

        self.private_key = None
        self.public_key = None
        self.load_keys()
        print("Blockchain initialization complete.")

    def create_genesis_block(self):
        return Block(0, time.time(), "Genesis Block", "0", "Genesis", "Genesis")  # Добавляем sender и signature

    def add_block(self, data):
        if self.private_key is None:
            raise ValueError("Private key not loaded. Generate or load keys first.")
        sender = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        signature = self.sign_data(data)

        previous_block = self.chain[-1]
        new_block_index = previous_block.index + 1
        new_block = Block(new_block_index, time.time(), data, previous_block.hash, sender, signature) # Добавляем sender и signature
        self.chain.append(new_block)
        self.save_chain()
        return new_block.hash

    def is_chain_valid(self):
        print("is_chain_valid called")
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            print(f"Checking block {i}")

            # Проверка хеша текущего блока
            calculated_hash = current_block.calculate_hash()
            if current_block.hash != calculated_hash:
                print(f"Hash mismatch at block {i}")
                print(f"  Calculated hash: {calculated_hash}")
                print(f"  Stored hash: {current_block.hash}")
                return False

            # Проверка хеша предыдущего блока
            if current_block.previous_hash != previous_block.hash:
                print(f"Previous hash mismatch at block {i}")
                print(f"  Current block's previous_hash: {current_block.previous_hash}")
                print(f"  Previous block's hash: {previous_block.hash}")
                return False

            # Проверка подписи
            if not self.verify_signature(current_block.data, current_block.signature, current_block.sender):
                print(f"Signature verification failed at block {i}")
                return False

        print("Blockchain is valid")
        return True

    def load_chain(self):
        print("load_chain called")
        try:
            with open(self.chain_file, 'r') as f:
                print("File opened successfully")
                chain_data = json.load(f)
                print(f"Loaded chain {chain_data}")
                chain = [Block.from_dict(block_data) for block_data in chain_data]
                print(f"Loaded chain: {chain}")
                return chain
        except FileNotFoundError:
            print("Blockchain file not found")
            return None
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return None
        except Exception as e:
            print(f"General exception in load_chain: {e}")
            return None
        finally:
            print("load_chain finished")

    def save_chain(self):
        print("save_chain called")  # Добавлено сообщение
        chain_data = [block.to_dict() for block in self.chain]
        with open(self.chain_file, 'w') as f:
            json.dump(chain_data, f, indent=4)

    def load_keys(self):
        try:
            with open("private_key.pem", "rb") as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
            with open("public_key.pem", "rb") as key_file:
                self.public_key = serialization.load_pem_public_key(
                    key_file.read(),
                    backend=default_backend()
                )
            print("Keys loaded successfully")  # Добавлено сообщение
        except FileNotFoundError:
            print("Key files not found, generating new keys.")
            self.generate_keys()

    def generate_keys(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.save_keys()
        print("Keys generated successfully")  # Добавлено сообщение

    def save_keys(self):
        if self.private_key is not None and self.public_key is not None:
            with open("private_key.pem", "wb") as key_file:
                key_file.write(self.private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            with open("public_key.pem", "wb") as key_file:
                key_file.write(self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
            print("Keys saved to private_key.pem and public_key.pem")
        else:
            print("Keys not generated. Call generate_keys() first.")

    def sign_data(self, data):
        if self.private_key is None:
            raise ValueError("Private key not loaded.")

        message = data.encode('utf-8')
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature.hex()

    def verify_signature(self, data, signature, public_key_pem):
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode('utf-8'),
                backend=default_backend()
            )

            signature_bytes = bytes.fromhex(signature)
            public_key.verify(
                signature_bytes,
                data.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False