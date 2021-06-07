
import csv
import psycopg2
import unicodedata

def strip_accent(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def strip_accents(string, accents=('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT', 'COMBINING TILDE')):
    accents = set(map(unicodedata.lookup, accents))
    chars = [c for c in unicodedata.normalize('NFD', string) if c not in accents]
    return unicodedata.normalize('NFC', ''.join(chars))

arr=[]
conn = psycopg2.connect(database="postgres", user="postgres", password="as", host="127.0.0.1", port="5432")
cur = conn.cursor()


inp = open("C:\\Users\AJESH\Desktop\ml-1m/movies1.dat", "r")
for line in inp.readlines():
    i = line.split('\t')
    a=int(i[0])
    b=str(i[1])
    c=i[4]
    b=b.decode('utf-8')

    cur.execute("INSERT INTO movie (idd, name, link) VALUES (%s,%s,%s)", (a-1,str(b),str(c)))




#
# cur.execute("SELECT id, name, link  from movie")
# rows = cur.fetchall()
# for row in rows:
#    print "ID = ", row[0]
#    print "NAME = ", row[1]
#    print "link = ", row[2], "\n"
#
# print "Operation done successfully";
#
#
#








conn.commit()
conn.close()