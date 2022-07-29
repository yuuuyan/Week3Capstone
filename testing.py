
import embed_captions
import embed_images
import organizeCOCO as COCO
import database as db

model = embed_images.train_model()
k = 4 #top k images
all_image_ids = COCO.return_ids() #i think this is what its supposed to be?
img_imbeds = db.database(all_image_ids, model)
query = input("input your search")
index_top_images = db.query(query, img_imbeds)
db.display(index_top_images, k, all_image_ids)