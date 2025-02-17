import os
import cv2
import glob
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2
import glob
from tqdm import tqdm


"""Helper functions"""
def _extract_data(df_row):
    return df_row['Anon MRN'], df_row['Anon TCIA Study Date'], df_row['Anon Exam Description'], df_row['Anon Study UID']

def load_ricord_metadata(ricord_meta_file):
    df = pd.read_excel(ricord_meta_file, sheet_name='CR Pos - TCIA Submission')
    ricord_metadata = []
    for index, row in df.iterrows():
        ricord_metadata.append(_extract_data(row))
    return ricord_metadata

def make_ricord_dict(ricord_data_set_file):
    """Loads bboxes from the given text file"""
    ricord_dict = {}
    with open(ricord_data_set_file, 'r') as f:
        for line in f.readlines():
            # Values after file name are crop dimensions
            if(len(line.split()) > 1):
                fname, xmin, ymin, xmax, ymax = line.rstrip('\n').split()
                bbox = tuple(int(c) for c in (xmin, ymin, xmax, ymax))
                ricord_dict[fname] = bbox
            else:
                fname = line.rstrip('\n')
                ricord_dict[fname] = None
                
    return ricord_dict

"""
RICORD data requires some preprocessing before splitting into test/train. 
Some images contain padding and some images are unusable. 
ricord_data_set.txt contains the name of usable images along with bounding box dimensions if needed. 

This cell crops the DICOM according to dimensions in ricord_data_set.txt and saves the image 
as png format in out_dir.

DICOM_images and "MIDRC-RICORD-1c Clinical Data Jan 13 2021 .xlsx" need to be downloaded from
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230281 before running this cell
"""

ricord_dir = '/MULTIX/DATA/HOME/ricord_datafiles'
ricord_meta_file = '/MULTIX/DATA/HOME/ricord_datafiles/MIDRC-RICORD-1c Clinical Data Jan 13 2021 .xlsx'

out_dir = 'ricord_images'
ricord_set_file = '/MULTIX/DATA/HOME/new_ricord_text.txt'

os.makedirs(out_dir, exist_ok=True)

# ricord_dict = make_ricord_dict(ricord_set_file)
# print(ricord_dict)

# metadata = load_ricord_metadata(ricord_meta_file)
# print(metadata)

# file_count = 0
# for mrn, date, desc, uid in metadata:
#     date = date.strftime('%m-%d-%Y')
#     uid = uid[-5:]
#     study_dir = os.path.join('/MULTIX/DATA/HOME/ricord_datafiles/ricord_data/MIDRC-RICORD-1C/MIDRC-RICORD-1C-{}'.format(mrn), '*-{}'.format(uid))
#     print(sorted(glob.glob(os.path.join(study_dir, '*', '*.dcm'))))
#     dcm_files = sorted(glob.glob(os.path.join(study_dir, '*', '*.dcm')))
#     # print(dcm_files)

#     for i, dcm_file in enumerate(dcm_files):
#         # Create output path and check if image is to be included
#         out_fname = 'MIDRC-RICORD-1C-{}-{}-{}.png'.format(mrn, uid, i)
#         out_path = os.path.join(out_dir, out_fname)
#         print(out_path)

#         if out_fname not in ricord_dict:
#             continue

#         # Load DICOM image
#         ds = pydicom.dcmread(dcm_file)
#         print(ds)

#         # Verify orientation
#         if ds.ViewPosition != 'AP' and ds.ViewPosition != 'PA':
#             print('Image from MRN-{} Date-{} UID-{} in position {}'.format(mrn, date, uid, ds.ViewPosition))
#             continue

#         # Apply transformations if required
#         if ds.pixel_array.dtype != np.uint8:
#              # Apply LUT transforms
#             arr = apply_modality_lut(ds.pixel_array, ds)
#             if arr.dtype == np.float64 and ds.RescaleSlope == 1 and ds.RescaleIntercept == 0:
#                 arr = arr.astype(np.uint16)
#             arr = apply_voi_lut(arr, ds)
#             arr = arr.astype(np.float64)

#             # Normalize to [0, 1]
#             arr = (arr - arr.min())/arr.ptp()

#             # Invert MONOCHROME1 images
#             if ds.PhotometricInterpretation == 'MONOCHROME1':
#                 arr = 1. - arr

#             # Convert to uint8
#             image = np.uint8(255.*arr)
#         else:
#             # Invert MONOCHROME1 images
#             if ds.PhotometricInterpretation == 'MONOCHROME1':
#                 image = 255 - ds.pixel_array
#             else:
#                 image = ds.pixel_array

#         # Crop if necessary
#         bbox = ricord_dict[out_fname]
#         if bbox is not None:
#             image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

#         # Save image
#         cv2.imwrite(out_path, image)
#         file_count += 1

# print('Created {} files'.format(file_count))

# set parameters here
savepath = '/home/ubuntu/frankenstein_data'
seed = 0
np.random.seed(seed) # Reset the seed so all runs are the same.
random.seed(seed)
MAXVAL = 255  # Range [0 255]

# path to covid-19 dataset from https://github.com/ieee8023/covid-chestxray-dataset
cohen_imgpath = '/MULTIX/DATA/HOME/covid-chestxray-dataset/images' 
cohen_csvpath = '/MULTIX/DATA/HOME/covid-chestxray-dataset/metadata.csv'

# path to covid-19 dataset from https://github.com/agchung/Figure1-COVID-chestxray-dataset
fig1_imgpath = '../Figure1-COVID-chestxray-dataset/images'
fig1_csvpath = '../Figure1-COVID-chestxray-dataset/metadata.csv'

# path to covid-19 dataset from https://github.com/agchung/Actualmed-COVID-chestxray-dataset
actmed_imgpath = '../Actualmed-COVID-chestxray-dataset/images'
actmed_csvpath = '../Actualmed-COVID-chestxray-dataset/metadata.csv'

# path to covid-19 dataset from https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
sirm_imgpath = '/home/ubuntu/COVID-19_Radiography_Dataset/COVID'
sirm_csvpath = '/home/ubuntu/COVID-19_Radiography_Dataset/COVID.metadata.xlsx'

# path to https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
rsna_datapath = ''
# get all the normal from here
rsna_csvname = '/MULTIX/DATA/HOME/stage_2_detailed_class_info.csv' 
# get all the 1s from here since 1 indicate pneumonia
# found that images that aren't pneunmonia and also not normal are classified as 0s
rsna_csvname2 = '/home/ubuntu/stage_2_train_labels.csv' 
rsna_imgpath = '/home/ubuntu/stage_2_train_images'

# path to ricord covid-19 images created by create_ricord_dataset/create_ricord_dataset.ipynb
# run create_ricord_dataset.ipynb before this notebook
ricord_imgpath = '/home/ubuntu/ricord_images'
ricord_txt = ricord_set_file

# parameters for COVIDx dataset
train = []
test = []

full_data = []

test_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
train_count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}

mapping = dict()
mapping['COVID-19'] = 'COVID-19'
mapping['SARS'] = 'pneumonia'
mapping['MERS'] = 'pneumonia'
mapping['Streptococcus'] = 'pneumonia'
mapping['Klebsiella'] = 'pneumonia'
mapping['Chlamydophila'] = 'pneumonia'
mapping['Legionella'] = 'pneumonia'
mapping['E.Coli'] = 'pneumonia'
mapping['Normal'] = 'normal'
mapping['Lung Opacity'] = 'pneumonia'
mapping['1'] = 'pneumonia'

# train/test split
split = 0.1
# to avoid duplicates
patient_imgpath = {}

# adapted from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py#L814
cohen_csv = pd.read_csv(cohen_csvpath, nrows=None)
#idx_pa = csv["view"] == "PA"  # Keep only the PA view
views = ["PA", "AP", "AP Supine", "AP semi erect", "AP erect"]
cohen_idx_keep = cohen_csv.view.isin(views)
cohen_csv = cohen_csv[cohen_idx_keep]

# fig1_csv = pd.read_csv(fig1_csvpath, encoding='ISO-8859-1', nrows=None)
# actmed_csv = pd.read_csv(actmed_csvpath, nrows=None)

sirm_csv = pd.read_excel(sirm_csvpath)


# get non-COVID19 viral, bacteria, and COVID-19 infections from covid-chestxray-dataset, figure1 and actualmed
# stored as patient id, image filename and label
filename_label = {'normal': [], 'pneumonia': [], 'COVID-19': []}
count = {'normal': 0, 'pneumonia': 0, 'COVID-19': 0}
covid_ds = {'cohen': [], 'fig1': [], 'actmed': [], 'sirm': [], 'ricord': []}

for index, row in cohen_csv.iterrows():
    f = row['finding'].split('/')[-1] # take final finding in hierarchy, for the case of COVID-19, ARDS
    if f == 'COVID-19' and ('eurorad.org' in row['url'] or 'ml-workgroup' in row['url'] or 'sirm.org' in row['url']):
        # skip COVID-19 positive images from eurorad to not duplicate sirm images
        pass
    elif f in mapping: # 
        count[mapping[f]] += 1
        entry = [str(row['patientid']), row['filename'], mapping[f], 'cohen']
        filename_label[mapping[f]].append(entry)
        if mapping[f] == 'COVID-19':
            covid_ds['cohen'].append(str(row['patientid']))
        
    
sirm = set(sirm_csv['URL'])
cohen = set(cohen_csv['url'])
cohen.add('https://github.com/ieee8023/covid-chestxray-dataset')  # Add base URL to remove sirm images from ieee dataset
discard = ['100', '101', '102', '103', '104', '105', 
           '110', '111', '112', '113', '122', '123', 
           '124', '125', '126', '217']

for idx, row in sirm_csv.iterrows():
    patientid = row['FILE NAME']
    if row['URL'] not in cohen and patientid[patientid.find('(')+1:patientid.find(')')] not in discard:
        count[mapping['COVID-19']] += 1
        imagename = patientid + '.' + row['FORMAT'].lower()
        if not os.path.exists(os.path.join(sirm_imgpath, imagename)):
            imagename = "COVID ({}).png".format(imagename.rsplit(".png")[0].split("COVID ")[1])
        entry = [patientid, imagename, mapping['COVID-19'], 'sirm']
        filename_label[mapping['COVID-19']].append(entry)
        covid_ds['sirm'].append(patientid)

# get ricord file names 
with open(ricord_txt) as f:
    ricord_file_names = [line.split()[0] for line in f]

for imagename in ricord_file_names:
    count[mapping['COVID-19']] += 1 # since RICORD data is all COVID-19 postive images
    print(imagename)
    patientid = imagename.split('-')[3] + '-' + imagename.split('-')[4]
    if patientid != 'MIDRC-RICORD-1C-419639-000025-04760-0':
        entry = [patientid, imagename, mapping['COVID-19'], 'ricord']
        filename_label[mapping['COVID-19']].append(entry)
    
        covid_ds['ricord'].append(patientid)
    
print('Data distribution from covid datasets:')
print(count)

#  Create list of RICORD patients to be added to test, equal to 200 images
# We want to prevent patients present in both train and test
# Get list of patients who have one image
ricord_patients = []
for label in filename_label['COVID-19']:
    if label[3] == 'ricord':
        ricord_patients.append(label[0])

pt_with_one_image = [x for x in ricord_patients if ricord_patients.count(x) == 1] # contrains 176 patients



# add covid-chestxray-dataset, figure1 and actualmed into COVIDx dataset
# since these datasets don't have test dataset, split into train/test by patientid
# for covid-chestxray-dataset:
# patient 8 is used as non-COVID19 viral test
# patient 31 is used as bacterial test
# patients 19, 20, 36, 42, 86 are used as COVID-19 viral test
# for figure 1:
# patients 24, 25, 27, 29, 30, 32, 33, 36, 37, 38

ds_imgpath = {'cohen': cohen_imgpath, 'sirm': sirm_imgpath, 'ricord':ricord_imgpath}

for key in filename_label.keys():
    arr = np.array(filename_label[key])
    if arr.size == 0:
        continue
    # split by patients
    # num_diff_patients = len(np.unique(arr[:,0]))
    # num_test = max(1, round(split*num_diff_patients))
    # select num_test number of random patients
    # random.sample(list(arr[:,0]), num_test)
    if key == 'pneumonia':
        test_patients = ['8', '31']
    elif key == 'COVID-19':
        test_patients = ['19', '20', '36', '42', '86', 
                         '94', '97', '117', '132', 
                         '138', '144', '150', '163', '169', '174', '175', '179', '190', '191',
                         'COVID-00024', 'COVID-00025', 'COVID-00026', 'COVID-00027', 'COVID-00029',
                         'COVID-00030', 'COVID-00032', 'COVID-00033', 'COVID-00035', 'COVID-00036',
                         'COVID-00037', 'COVID-00038',
                         'ANON24', 'ANON45', 'ANON126', 'ANON106', 'ANON67',
                         'ANON153', 'ANON135', 'ANON44', 'ANON29', 'ANON201', 
                         'ANON191', 'ANON234', 'ANON110', 'ANON112', 'ANON73', 
                         'ANON220', 'ANON189', 'ANON30', 'ANON53', 'ANON46',
                         'ANON218', 'ANON240', 'ANON100', 'ANON237', 'ANON158',
                         'ANON174', 'ANON19', 'ANON195',
                         'COVID 119', 'COVID 87', 'COVID 70', 'COVID 94', 
                         'COVID 215', 'COVID 77', 'COVID 213', 'COVID 81', 
                         'COVID 216', 'COVID 72', 'COVID 106', 'COVID 131', 
                         'COVID 107', 'COVID 116', 'COVID 95', 'COVID 214', 
                         'COVID 129']
#         Add 178 RICORD patients to COVID-19, equal to 200 images
        test_patients.extend(pt_with_one_image)
        test_patients.extend(['419639-000025', '419639-001464'])
    else: 
        test_patients = []
    print('Key: ', key)
    print('Test patients: ', test_patients)
    # go through all the patients
    for patient in tqdm(arr):
        if patient[0] not in patient_imgpath:
            patient_imgpath[patient[0]] = [patient[1]]
        else:
            if patient[1] not in patient_imgpath[patient[0]]:
                patient_imgpath[patient[0]].append(patient[1])
            else:
                continue  # skip since image has already been written
        if patient[0] in test_patients:
            if patient[0] in ['419639-000025','419639-000002','419639-000082','419639-000086','419639-000235','419639-000299']:
                print('pass', patient[0])
                pass
            else:
                if patient[3] == 'sirm':
                    image = cv2.imread(os.path.join(ds_imgpath[patient[3]], patient[1]))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    patient[1] = patient[1].replace(' ', '')
                    cv2.imwrite(os.path.join(savepath, 'test', patient[1]), gray)
                else:
                    copyfile(os.path.join(ds_imgpath[patient[3]], patient[1]), os.path.join(savepath, 'test', patient[1]))
            
                test.append(patient)
                test_count[patient[2]] += 1
        else:
            if patient[0] in ['419639-000025','419639-000238','419639-000331','419639-000214','419639-000215','419639-000303']:
                print('pass', patient[0])
                pass
            else:
                if patient[3] == 'sirm':
                    image = cv2.imread(os.path.join(ds_imgpath[patient[3]], patient[1]))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    patient[1] = patient[1].replace(' ', '')
                    cv2.imwrite(os.path.join(savepath, 'train', patient[1]), gray)
                else:
                    copyfile(os.path.join(ds_imgpath[patient[3]], patient[1]), os.path.join(savepath, 'train', patient[1]))
                train.append(patient)
                train_count[patient[2]] += 1

print('test count: ', test_count)
print('train count: ', train_count)

# add normal and rest of pneumonia cases from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
csv_normal = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname), nrows=None)
csv_pneu = pd.read_csv(os.path.join(rsna_datapath, rsna_csvname2), nrows=None)
patients = {'normal': [], 'pneumonia': []}

for index, row in csv_normal.iterrows():
    if row['class'] == 'Normal':
        patients['normal'].append(row['patientId'])

for index, row in csv_pneu.iterrows():
    if int(row['Target']) == 1:
        patients['pneumonia'].append(row['patientId'])

for key in patients.keys():
    arr = np.array(patients[key])
    if arr.size == 0:
        continue
    # split by patients 
    # num_diff_patients = len(np.unique(arr))
    # num_test = max(1, round(split*num_diff_patients))
    test_patients = np.load('/MULTIX/DATA/HOME/rsna_test_patients_{}.npy'.format(key)) # random.sample(list(arr), num_test), download the .npy files from the repo.
    # np.save('rsna_test_patients_{}.npy'.format(key), np.array(test_patients))
    for patient in arr:
        if patient not in patient_imgpath:
            patient_imgpath[patient] = [patient]
        else:
            continue  # skip since image has already been written
                
        ds = dicom.dcmread(os.path.join(rsna_datapath, rsna_imgpath, patient + '.dcm'))
        pixel_array_numpy = ds.pixel_array
        imgname = patient + '.png'
        if patient in test_patients:
            cv2.imwrite(os.path.join(savepath, 'test', imgname), pixel_array_numpy)
            entry = [patient, imgname, key, 'rsna']
            test.append(entry)
            test_count[key] += 1
        else:
            cv2.imwrite(os.path.join(savepath, 'train', imgname), pixel_array_numpy)
            entry = [patient, imgname, key, 'rsna']
            train.append(entry)

            train_count[key] += 1



train_df = pd.DataFrame(train)
train_df.columns = ["patientid", "img",'finding', "source"]
train_df['split'] = 'train'
print(train_df['source'].value_counts())

test_df = pd.DataFrame(test)
test_df.columns = ["patientid", "img",'finding', "source"]
test_df['split'] = 'test'
print(test_df['source'].value_counts())

full_data = pd.concat([train_df, test_df])

test_sample = full_data.sample(frac=0.1)
test_sample['source_split'] = 'test'
print(test_sample['source'].value_counts())


train_sample = full_data.drop(test_sample)
train_sample['source_split'] = 'train'
print(train_sample['source'].value_counts())

full_data = pd.concat([test_sample, train_sample])

full_data.to_csv('/MULTIX/DATA/HOME/frankenstein_data.csv')

