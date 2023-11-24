gdown "1Jm67b342FtyFfFvLAvWH-hO-WOLpvNg4"
gdown "1MRJpIfn8904qJ00BXuiRm4N4KB3fXKCu"
unzip pitch.zip -d ./data/
unzip energy.zip -d ./data/
mv ./data/dla/data/pitch ./data/
mv ./data/dla/data/energy ./data/
rm -rf ./data/dla
rm pitch.zip
rm energy.zip