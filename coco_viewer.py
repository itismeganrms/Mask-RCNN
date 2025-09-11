from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2

# Path to your COCO-style JSON annotation file
annFile = "/home/mrajaraman/do-not-modify/image-to-coco-json-converter/output/test.json"
coco = COCO(annFile)
print("Loading annotations")
# Pick an image
imgId = coco.getImgIds()[0]
imgInfo = coco.loadImgs(imgId)[0]
image_path = f"/home/mrajaraman/dataset/coco/test/img_1458493990.png"
print("Image path saved successfully")
# Load image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load annotations
ann_ids = coco.getAnnIds(imgIds=0)
anns = coco.loadAnns(ann_ids)

plt.imshow(image)
coco.showAnns(anns)

# Add ann IDs
for ann in anns:
    bbox = ann["bbox"]  # [x, y, w, h]
    ann_id = ann["id"]
    x, y = bbox[0], bbox[1]  # place text at top-left of the bbox
    plt.text(
        x, y - 2, str(ann_id),
        color="red", fontsize=10, weight="bold", backgroundcolor="white"
    )

plt.axis("off")
plt.savefig("vis_with_ids_new.png", bbox_inches="tight", pad_inches=0)
plt.close()