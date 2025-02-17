import os

!git clone https://github.com/ieee8023/covid-chestxray-dataset.git
!git clone https://github.com/agchung/Figure1-COVID-chestxray-dataset.git
!git clone https://github.com/agchung/Actualmed-COVID-chestxray-dataset.git

!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6

os.environ['KAGGLE_USERNAME']="rachaelharkness"
os.environ['KAGGLE_KEY']="8b5f5da7eed94627088c8978864d9078"

!kaggle competitions download -c rsna-pneumonia-detection-challenge
!unzip rsna-pneumonia-detection-challenge.zip

!kaggle datasets download -d tawsifurrahman/covid19-radiography-database
!unzip covid19-radiography-database.zip