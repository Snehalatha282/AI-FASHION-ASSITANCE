import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
from PIL import Image , ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


dataset_folder = "Data"
chroma_client = chromadb.PersistentClient(path="Vector_database")
image_loader = ImageLoader()
CLIP = OpenCLIPEmbeddingFunction()

image_vib = chroma_client.get_or_create_collection(name="image", embedding_function = CLIP,data_loader=image_loader)

ids = []
uris= []

for i , filename in enumerate(sorted(os.listdir(dataset_folder))):
   if filename.lower().endswith('.png'):
    file_path = os.path.join(dataset_folder, filename)
    


    try:
      with Image.open(file_path) as img:
        img.verify()  # check integrity
      ids.append(str(i))
      uris.append(file_path)
    except Exception as e:
        print(f"⚠️ Skipping {filename}: {e}")

# Only add if we have valid images
if ids:
    image_vib.add(ids=ids, uris=uris)
    print(f"✅ Stored {len(ids)} images to the vector database.")
else:
    print("❌ No valid images found.")

   #ids.append(str(i))
    #uris.append(file_path)

#image_vib.add(
 # ids=ids,
  #uris=uris

#)
#print("Images stored to the vector database.")