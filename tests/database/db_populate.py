import csv
import uuid
import psycopg2
import ast
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from itertools import islice


pg_conn = psycopg2.connect(
    dbname="db",
    user="user",
    password="password",
    host="10",
    port="5432"
)
pg_cur = pg_conn.cursor()


# write logic of inserting data in bd 






# Clean up
pg_cur.close()
pg_conn.close()