
import embed_captions
import embed_images
import organizeCOCO
import database as db

k = 4 #top k images
img_imbeds = db.database(     )
query = input("input your search")
id_top_images = db.query(query, img_imbeds)
db.display(id_top_images, k,     )