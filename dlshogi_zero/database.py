import numpy as np
import sqlite3
import os.path
from cshogi import *

HcpAndRepetition = np.dtype([
    ('hcp', dtypeHcp),
    ('repetition', np.int8)
    ])

dtypeVisit = np.dtype(np.int16)

COMMIT_INTERVAL = 100

class TrainingDataBase:
    def __init__(self, filepath, clear=False):
        file_exist = os.path.exists(filepath)
        self.con = sqlite3.connect(filepath, check_same_thread=False)
        self.cur = self.con.cursor()
        self.model_ver = 0
        if file_exist and clear:
            self.cur.execute('DROP TABLE training_data')
        if not file_exist or clear:
            self.cur.execute('CREATE TABLE training_data (game_id integer, model_ver integer, hcprs blob, total_move_count integer, legal_moves blob, visits blob, game_result integer)')
            self.cur.execute('CREATE INDEX idx_game_id ON training_data (game_id)')
            self.current_game_id = 0
        else:
            self.current_game_id = (self.cur.execute('SELECT MAX(game_id) FROM training_data').fetchone()[0] or -1) + 1

    def close(self):
        self.con.commit()
        self.cur.close()
        self.con.close()
    
    def set_model_ver(self, model_ver):
        self.model_ver = model_ver

    def write_chunk(self, chunk):
        self.cur.executemany('INSERT INTO training_data VALUES ({}, {}, ?, ?, ?, ?, ?)'.format(self.current_game_id, self.model_ver), chunk)
        self.current_game_id += 1
        if self.current_game_id % COMMIT_INTERVAL == 0:
            self.con.commit()

    def prepare_training(self, window_size):
        self.max_row_id = self.cur.execute('SELECT MAX(rowid) FROM training_data').fetchone()[0]
        self.min_row_id = self.cur.execute('SELECT MIN(rowid) FROM training_data WHERE game_id >= {min_game_id}'.format(min_game_id=self.current_game_id - window_size)).fetchone()[0]

    def get_training_batch(self, batch_size):
        batch = self.cur.execute(
            'WITH RECURSIVE rand_id(id) AS (SELECT RANDOM() UNION ALL SELECT RANDOM() FROM rand_id LIMIT {n_samples}) SELECT hcprs, total_move_count, legal_moves, visits, game_result FROM training_data AS A JOIN (SELECT ABS(id) % ({max_row_id} - {min_row_id}) + {min_row_id} AS B FROM rand_id) ON A.rowid = B'.format(
                n_samples=batch_size,
                max_row_id=self.max_row_id,
                min_row_id=self.min_row_id)).fetchall()
        for data in batch:
            yield (
                np.frombuffer(data[0], dtype=HcpAndRepetition),
                data[1],
                np.frombuffer(data[2], dtype=dtypeMove16),
                np.frombuffer(data[3], dtype=dtypeVisit),
                data[4]
                )
