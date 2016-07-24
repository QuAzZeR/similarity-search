import sqlite3
conn = sqlite3.connect('database_similarity.db')
c = conn.cursor()
# c.execute('''CREATE TABLE cluster_word
#              (center_word text, max real, min real, similarity_word text)''')
c.execute("SELECT * FROM cluter_word")
print(c.fetchall())
conn.commit()

conn.close()
