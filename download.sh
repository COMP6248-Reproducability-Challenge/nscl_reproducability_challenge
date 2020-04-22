mkdir -p data
cd data

echo "Downloading CLEVR dataset..."
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
rm CLEVR_v1.0.zip

echo "Downloading annotated scene"
cd CLEVR_v1.0/scenes
wget http://nscl.csail.mit.edu/data/code-data/clevr/train/scenes.json.zip
unzip scenes.json.zip -d train
rm scenes.json.zip

wget http://nscl.csail.mit.edu/data/code-data/clevr/val/scenes.json.zip
unzip scenes.json.zip -d val
rm scenes.json.zip

echo "Downloading pretrained model..."
wget http://nsvqa.csail.mit.edu/assets/pretrained.zip
unzip pretrained.zip
rm pretrained.zip

mkdir mask_rcnn
cd mask_rcnn
echo "Downloading pretrained weights for Mask R-CNN backbone..."
wget http://nsvqa.csail.mit.edu/assets/backbones.zip
unzip backbones.zip
rm backbones.zip

cd ..  # root