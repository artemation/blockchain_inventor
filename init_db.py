import os
import psycopg2

# Установка соединения с БД
conn = psycopg2.connect(host="amvera-artemation-cnpg-blockchain-invent-rw",
                        database=os.environ['POSTGRES_DB'],
                        user=os.environ['USERNAME_DB'],
                        password=os.environ['PASSWORD_DB'])

# Курсор - это объект, позволяющий взаимодействовать с БД
cur = conn.cursor()

# Создание таблицы blockchain_blocks
cur.execute('DROP TABLE IF EXISTS blockchain_blocks;')
cur.execute('CREATE TABLE blockchain_blocks ('
            'id serial PRIMARY KEY,'
            'index integer NOT NULL,'
            '"timestamp" timestamp without time zone NOT NULL,'
            'transactions text NOT NULL,'
            'previous_hash char(64) NOT NULL,'
            'hash char(64) NOT NULL UNIQUE,'
            'node_id integer NOT NULL);'
            )

# Создание таблицы users
cur.execute('DROP TABLE IF EXISTS users;')
cur.execute('CREATE TABLE users ('
            'id serial PRIMARY KEY,'
            'username varchar(255),'
            'password varchar(255),'
            'is_admin boolean,'
            'role varchar(50));'
            )

# Создание таблицы invitations
cur.execute('DROP TABLE IF EXISTS invitations;')
cur.execute('CREATE TABLE invitations ('
            'id serial PRIMARY KEY,'
            'code varchar(255) NOT NULL UNIQUE,'
            'email varchar(255),'
            'created_at timestamp DEFAULT CURRENT_TIMESTAMP,'
            'user_id integer REFERENCES users(id),'
            'used_at timestamp);'
            )

# Создание таблицы Единица_измерения
cur.execute('DROP TABLE IF EXISTS "Единица_измерения";')
cur.execute('CREATE TABLE "Единица_измерения" ('
            '"Единица_ИзмеренияID" serial PRIMARY KEY,'
            '"Единица_Измерения" varchar(255));'
            )

# Создание таблицы Товары
cur.execute('DROP TABLE IF EXISTS "Товары";')
cur.execute('CREATE TABLE "Товары" ('
            '"ТоварID" serial PRIMARY KEY,'
            '"Наименование" varchar(255),'
            '"Описание" text,'
            '"Единица_ИзмеренияID" integer REFERENCES "Единица_измерения"("Единица_ИзмеренияID"));'
            )

# Создание таблицы Склады
cur.execute('DROP TABLE IF EXISTS "Склады";')
cur.execute('CREATE TABLE "Склады" ('
            '"СкладID" serial PRIMARY KEY,'
            '"Название" varchar(255),'
            '"Адрес" varchar(255),'
            '"Телефон" varchar(20));'
            )

# Создание таблицы Тип_документа
cur.execute('DROP TABLE IF EXISTS "Тип_документа";')
cur.execute('CREATE TABLE "Тип_документа" ('
            '"ДокументID" serial PRIMARY KEY,'
            '"Тип_документа" varchar(255));'
            )

# Создание таблицы Запасы
cur.execute('DROP TABLE IF EXISTS "Запасы";')
cur.execute('CREATE TABLE "Запасы" ('
            '"ЗапасID" serial PRIMARY KEY,'
            '"ТоварID" integer REFERENCES "Товары"("ТоварID"),'
            '"СкладID" integer REFERENCES "Склады"("СкладID"),'
            '"Количество" real,'
            '"Дата_обновления" date,'
            '"Единица_ИзмеренияID" integer REFERENCES "Единица_измерения"("Единица_ИзмеренияID"));'
            )

# Создание таблицы ПриходРасход
cur.execute('DROP TABLE IF EXISTS "ПриходРасход";')
cur.execute('CREATE TABLE "ПриходРасход" ('
            '"ПриходРасходID" serial PRIMARY KEY,'
            '"СкладОтправительID" integer REFERENCES "Склады"("СкладID"),'
            '"ДокументID" integer REFERENCES "Тип_документа"("ДокументID"),'
            '"ТоварID" integer REFERENCES "Товары"("ТоварID"),'
            '"Количество" real,'
            '"Единица_ИзмеренияID" integer REFERENCES "Единица_измерения"("Единица_ИзмеренияID"),'
            '"СкладПолучательID" integer REFERENCES "Склады"("СкладID"),'
            '"TransactionHash" varchar,'
            '"Timestamp" varchar);'
            )

# Вставка данных в таблицу Единица_измерения
cur.execute('INSERT INTO "Единица_измерения" ("Единица_ИзмеренияID", "Единица_Измерения") '
            'VALUES (%s, %s), (%s, %s), (%s, %s)',
            (3, 'Килограмм', 2, 'Штука', 1, 'Литр'))

# Вставка данных в таблицу users
cur.execute('INSERT INTO users (id, username, password, is_admin, role) '
            'VALUES (%s, %s, %s, %s, %s), (%s, %s, %s, %s, %s)',
            (1, 'admin', 'admin123', True, 'admin',
             2, 'user1', 'user123', False, 'user'))

# Вставка данных в таблицу invitations
cur.execute('INSERT INTO invitations (id, code, email, user_id) '
            'VALUES (%s, %s, %s, %s)',
            (1, 'CODE123', 'user1@example.com', 1))

# Вставка данных в таблицу Склады
cur.execute('INSERT INTO "Склады" ("СкладID", "Название", "Адрес", "Телефон") '
            'VALUES (%s, %s, %s, %s), (%s, %s, %s, %s)',
            (1, 'Склад 1', 'Адрес склада 1', '1234567890',
             2, 'Склад 2', 'Адрес склада 2', '0987654321'))

# Вставка данных в таблицу Тип_документа
cur.execute('INSERT INTO "Тип_документа" ("ДокументID", "Тип_документа") '
            'VALUES (%s, %s), (%s, %s)',
            (1, 'Приход', 2, 'Расход'))

# Вставка данных в таблицу Товары
cur.execute('INSERT INTO "Товары" ("ТоварID", "Наименование", "Единица_ИзмеренияID") '
            'VALUES (%s, %s, %s), (%s, %s, %s)',
            (1, 'Товар 1', 2, 2, 'Товар 2', 1))

# Вставка данных в таблицу Запасы
cur.execute('INSERT INTO "Запасы" ("ЗапасID", "ТоварID", "СкладID", "Количество", "Дата_обновления", "Единица_ИзмеренияID") '
            'VALUES (%s, %s, %s, %s, %s, %s), (%s, %s, %s, %s, %s, %s)',
            (1, 1, 1, 100.0, '2023-01-01', 2,
             2, 2, 2, 50.5, '2023-01-02', 1))

# Вставка данных в таблицу ПриходРасход
cur.execute('INSERT INTO "ПриходРасход" ("ПриходРасходID", "СкладОтправительID", "ДокументID", "ТоварID", "Количество", "Единица_ИзмеренияID", "СкладПолучательID") '
            'VALUES (%s, %s, %s, %s, %s, %s, %s), (%s, %s, %s, %s, %s, %s, %s)',
            (1, 1, 1, 1, 10.0, 2, 2,
             2, 2, 2, 2, 5.5, 1, 1))

# Вставка данных в таблицу blockchain_blocks
cur.execute('INSERT INTO blockchain_blocks (id, index, "timestamp", transactions, previous_hash, hash, node_id) '
            'VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (1, 1, '2023-01-01 10:00:00', 'transactions1', 'prevhash1', 'hash1', 1))

# Установка значений последовательностей
cur.execute("SELECT setval('blockchain_blocks_id_seq', 16, true)")
cur.execute("SELECT setval('invitations_id_seq', 1, true)")
cur.execute("SELECT setval('users_id_seq', 3, true)")

# Сохранение операций
conn.commit()

# Закрытие соединения
cur.close()
conn.close()