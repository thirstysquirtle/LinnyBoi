import pymongo


client = pymongo.MongoClient("mongodb+srv://bob:w1w2w3w4@freetier.lenkq.mongodb.net/?retryWrites=true&w=majority")
db = client.test

# print(db["test"])