import numpy as np
from scipy.ndimage import zoom
from SimpleITK import ReadImage
import SimpleITK as sitk
import glob
from standardUtils import multi_slice_viewer
import os
import pandas as pd
from standardUtils import multi_slice_viewer
import torch
from unet import UNet
from skimage.measure import label, regionprops_table
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

file = "multi_scan_split.xlsx"
final_resolution = (1,1,1)
final_size = (448, 448, 448)
filetype = "*.nii"
savename = "E:/Brennan/KITS/kca_cohortfinder_multiscan_all_segs.h5"
clinical_files = ["E:/Brennan/TCGA_KIRC/KIRC_Clinical_info_cleaned.xlsx",
            "E:/Brennan/TCGA_KICH/KICH_Clinical_Info cleaned.xlsx",
            "E:/Brennan/TCGA_KIRP/KIRP Clinical Data Cleaned.xlsx",
            "E:/Brennan/kits23-v0.1.2/kits23_clinical_renamed.csv"]
modelname = "ktseg_kits2023_best_model.pth"
extra_files = "E:\Brennan\kits23-v0.1.2\kits23\dataset"
extra_type = "Folder"
fix_events = False
fix_events_group = False
random_split = False
keep_segs = True
num_bins = 2

# Load segmentation model for kidney localization
torch.cuda.set_device(0)
device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
model = UNet(n_classes=3, in_channels=1,
             padding=True, depth=6, wf=4,
             up_mode='upconv', batch_norm=True).to(device)
checkpoint = torch.load(modelname, map_location=lambda storage,
                                                        loc: storage)
model.load_state_dict(checkpoint["model_dict"])
model.eval()

# Kidney ROI Function
def get_volumes(img_stack):
    kidney_stack = np.zeros(np.shape(img_stack))
    tumor_stack = np.zeros(np.shape(img_stack))
    for i in range(np.shape(img_stack)[0]):
        input = torch.Tensor(img_stack[i,::].copy())[None,None,::].to(device)
        output = torch.softmax(model(input), dim=1)[0,::].permute(1,2,0).detach().cpu().numpy()
        kidney_stack[i,::] = output[:,:,0]
        tumor_stack[i,::] = output[:,:,1]

    com = (kidney_stack > .34)
    if np.any(com):
        kt_labeled = label(com)
        info = pd.DataFrame(regionprops_table(kt_labeled, properties=["centroid", "area"])).sort_values("area")
        if len(info) < 2:
            if keep_segs:
                return None, None, None
            else:
                return None, None
        info = info.iloc[-2:]# Take the largest segmentation, should be the tumor
        info = info.reset_index()

        # Check amount of tumor in each "kidney"
        tv = [0,0]
        for ii, (index, row) in enumerate(info.iterrows()):
            cen = [row["centroid-0"], row["centroid-1"], row["centroid-2"]]
            cen = [int(i) for i in cen]
            for i in range(2):
                cen[i] = cen[i] + (np.shape(img_stack)[i] - (cen[i] + 50)) if (cen[i] + 50) > np.shape(img_stack)[i] else cen[i]
                cen[i] = cen[i] - (cen[i] - 50) if (cen[i] - 50) < 0 else cen[i]
            tumor_temp = tumor_stack[(cen[0] - 50):(cen[0] + 50), (cen[1] - 50):(cen[1] + 50), (cen[2] - 50):(cen[2] + 50)]
            tv[ii] = np.sum((tumor_temp > .34) * 1)
        info["tumor_area"] = tv
        info = info.sort_values("tumor_area")
        vol_stack = np.zeros((len(info["centroid-0"]), 100,100, 100))
        seg_stack = np.zeros((len(info["centroid-0"]), 2,100,100, 100))
        model_output = (kidney_stack > .34) + (tumor_stack > .34) > 0 * 1
        for ii, (index, row) in enumerate(info.iterrows()):
            cen = [row["centroid-0"], row["centroid-1"], row["centroid-2"]]
            cen = [int(i) for i in cen]
            for i in range(3):
                cen[i] = cen[i] + (np.shape(img_stack)[i] - (cen[i] + 50)) if (cen[i] + 50) > np.shape(img_stack)[i] else cen[i]
                cen[i] = cen[i] - (cen[i] - 50) if (cen[i] - 50) < 0 else cen[i]

            temp_stack = img_stack[(cen[0] - 50):(cen[0] + 50), (cen[1] - 50):(cen[1] + 50), (cen[2] - 50):(cen[2] + 50)]
            temp_seg_stack = model_output[(cen[0] - 50):(cen[0] + 50), (cen[1] - 50):(cen[1] + 50), (cen[2] - 50):(cen[2] + 50)]
            vol_stack[ii, :, :, :] = temp_stack
            seg_stack[ii, 0, :,:,:] = (kidney_stack[(cen[0] - 50):(cen[0] + 50), (cen[1] - 50):(cen[1] + 50), (cen[2] - 50):(cen[2] + 50)] > .34) * 1
            seg_stack[ii, 1,:, :, :] = (tumor_stack[(cen[0] - 50):(cen[0] + 50), (cen[1] - 50):(cen[1] + 50), (cen[2] - 50):(cen[2] + 50)] > 0.34) * 1

        if keep_segs:
            return vol_stack[1], vol_stack[0], seg_stack[1]
        else:
            return vol_stack[1], vol_stack[0]
    else:
        if keep_segs:
            return None, None, None
        else:
            return None, None


# Load Clinical files
data = pd.read_excel(file)
clinical = []
for file in clinical_files:
    ci = pd.read_csv(file) if os.path.splitext(file)[-1] == ".csv" else pd.read_excel(file)
    clinical = pd.concat([clinical, ci],ignore_index=True) if len(clinical) > 0 else ci
    bla = 1

if fix_events:


    #remove patients that dont have clinical info
    data = data[data['patient_id'].isin(clinical['patient_id'])]
    data = data.reset_index(drop=True)
    data["events"] = [clinical.loc[clinical['patient_id'] == pid, 'vital_status'].values[[0]][0] for pid in
                            data.patient_id]
    data["event_times"] = [clinical.loc[clinical['patient_id'] == pid, 'vital_days_after_surgery'].values[[0]][0]
                                 for pid in data.patient_id]
    event_data = data[data.events == 1]
    non_event_data = data[data.events == 0]
    bins = np.linspace(start=min(event_data.event_times), stop=max(event_data.event_times),
                       num=10)  # 10 bins = 11 edges
    data["bin_indices"] = np.digitize(data.event_times, bins, right=False)
    datacopy = data.copy()

    plt.hist(
        [np.array(
            [x for i, x in enumerate(data["event_times"]) if data["testind"][i] == phase and data["events"][i] == 1])
         for phase
         in [0, 1, 2]],
        label=["train", "test", "val"], color=["r", "g", "b"], alpha=0.5, stacked=True, edgecolor='black')
    plt.title("Event times, event cases only, fixed")
    plt.legend()
    plt.show()

    plt.hist(
        [np.array([x for i, x in enumerate(data["event_times"]) if data["testind"][i] == phase]) for phase
         in [0, 1, 2]],
        label=["train", "test", "val"], color=["r", "g", "b"], alpha=0.5, stacked=True, edgecolor='black')
    plt.title("Event times, fixed")
    plt.legend()
    plt.show()

    if fix_events_group:
        groups = np.sort(data.groupid.unique())
    else:
        groups = [0]
    for group in groups:
        if fix_events_group:
            group_data = data[data.groupid == group]
        else:
            group_data = data.copy()
        # group_data["events"] = [clinical.loc[clinical['patient_id'] == pid, 'vital_status'].values[[0]][0]for pid in group_data.patient_id]
        # group_data["event_times"] = [clinical.loc[clinical['patient_id'] == pid, 'vital_days_after_surgery'].values[[0]][0] for pid in group_data.patient_id]
        for bin in np.sort(data["bin_indices"].unique()):
            bin_data_temp = group_data[group_data.bin_indices == bin]
            bin_data_events = bin_data_temp[bin_data_temp.events == 1]
            bin_data_non_events = bin_data_temp[bin_data_temp.events == 0]
            for bin_data in [bin_data_events, bin_data_non_events]:
                if len(bin_data) > 0:
                    if len(bin_data) == 1: # if there is only 1 event in the bin, assign it to train
                        data.loc[data.patient_id == bin_data.patient_id.item(), "testind"] = 0
                    elif len(bin_data) == 2: # if there is 2 events, assign to train and test
                        data.loc[data.patient_id == bin_data.iloc[0].patient_id, "testind"] = 0
                        data.loc[data.patient_id == bin_data.iloc[1].patient_id, "testind"] = 1
                    elif len(bin_data) == 3: # if there are 3 events, assign equally to all three
                        data.loc[data.patient_id == bin_data.iloc[0].patient_id, "testind"] = 0
                        data.loc[data.patient_id == bin_data.iloc[1].patient_id, "testind"] = 1
                        data.loc[data.patient_id == bin_data.iloc[2].patient_id, "testind"] = 2
                    else:
                        bin_data = bin_data.sample(frac=1) #shuffle data
                        for i in range(0, int(len(bin_data)*.5)): # training
                            data.loc[data.patient_id == bin_data.iloc[i].patient_id, "testind"] = 0
                        for i in range(round(len(bin_data) * .5), round(len(bin_data) * .75)):  # test
                            data.loc[data.patient_id == bin_data.iloc[i].patient_id, "testind"] = 1
                        for i in range(round(len(bin_data) * .75), len(bin_data)):  # val
                            data.loc[data.patient_id == bin_data.iloc[i].patient_id, "testind"] = 2

        split_dict = {0: "Train", 1: "Test", 2: "Val"}
        data["Split"] = [split_dict[x] for x in data["testind"].tolist()]

        plt.hist(
            [np.array([x for i, x in enumerate(data["event_times"]) if data["testind"][i] == phase and data["events"][i] == 1]) for phase
             in [0,1,2]],
            label=["train", "test", "val"], color=["r", "g", "b"], alpha=0.5, stacked=True, edgecolor='black')
        plt.title("Event times, event cases only, fixed")
        plt.legend()
        plt.show()

        plt.hist(
            [np.array([x for i, x in enumerate(data["event_times"]) if data["testind"][i] == phase]) for phase
             in [0,1,2]],
            label=["train", "test", "val"], color=["r", "g", "b"], alpha=0.5, stacked=True, edgecolor='black')
        plt.title("Event times, fixed")
        plt.legend()
        plt.show()

        bla = 1

def normalize_image(image):
    # Resample the image to the new spacing
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(final_resolution)
    resample.SetSize(final_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(-1024)

    # Apply the resampling
    new_image = resample.Execute(image)
    new_image_vol = sitk.GetArrayFromImage(new_image)

    return new_image, new_image_vol


# Assign class values for classification
def assign_bins(data, num_bins):
    # Sort the data
    sorted_data = np.sort(data)
    # Calculate the bin edges based on quantiles
    bin_edges = np.interp(
        np.linspace(0, len(sorted_data), num_bins + 1),
        np.arange(len(sorted_data)),
        sorted_data
    )
    return bin_edges

es = []
ts = []
for index, row in data.iterrows():
    if row.patient_id in clinical.patient_id.values:
        pat_clinical = clinical[clinical['patient_id'] == row.patient_id]
        e = pat_clinical.vital_status.item()
        t = pat_clinical.vital_days_after_surgery.item()
        es.append(e)
        ts.append(t)

ts_events = [t for t,e in zip(ts, es) if e==1]
cutoffs = np.array([1,2,2.5,3,3.5]) * 365
response_dict = {(True, 0): 0, (True,1): 0, (False, 0): np.nan, (False, 1): 1}
for cutoff in cutoffs:
    clinical[f"class_{cutoff}"] = [response_dict[(t > cutoff, e)] for t,e in
                                   zip(clinical.vital_days_after_surgery, clinical.vital_status)]

binedges = assign_bins(ts_events, num_bins)
# Assign bin values to each data point
bin_indices = np.digitize(clinical.vital_days_after_surgery, binedges, right=True)
# Adjust bin indices to make them 0-based and ensure the maximum index fits within the number of bins
clinical["EventBin"] = np.clip(bin_indices - 1, 0, num_bins - 1)


def assign_split(df, patient_id_col='patient_id'):
    # Get unique patient ids
    unique_patient_ids = df[patient_id_col].unique()

    # Shuffle the unique patient ids
    np.random.shuffle(unique_patient_ids)

    # Calculate the split sizes
    n_patients = len(unique_patient_ids)
    n_train = int(n_patients * 0.6)
    n_test = int(n_patients * 0.2)
    # The rest goes to validation (remaining patients)

    # Assign patients to splits
    train_ids = unique_patient_ids[:n_train]
    test_ids = unique_patient_ids[n_train:n_train + n_test]
    validation_ids = unique_patient_ids[n_train + n_test:]

    # Create a new column 'Split' and assign the appropriate group
    df['Split'] = np.where(df[patient_id_col].isin(train_ids), 'Train',
                           np.where(df[patient_id_col].isin(test_ids), 'Test', 'Val'))

    return df

if random_split:
    data = assign_split(data.copy())
    train_count = data['Split'].value_counts().get('Train', 0)
    test_count = data['Split'].value_counts().get('Test', 0)
    val_count = data['Split'].value_counts().get('Val', 0)

output = {x: {"x": np.array([]), "x2nd": np.array([]), "label": [], "e": [], "t": [], "name": [], "class_365.0":[], "class_730.0":[], "class_912.5":[], "class_1277.5":[]}
          for x in ["train", "test", "val", "exval"]}
if keep_segs:
    for subdict in output.values():
        subdict["seg"] = np.array([])

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Main Files"):
    if row.patient_id in clinical.patient_id.values:

        folder = row.Folder
        pat_clinical = clinical[clinical['patient_id'] == row.patient_id]
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        old_image_vol = sitk.GetArrayFromImage(image)

        new_image, new_image_vol = normalize_image(image)

        # Get Kidney + Tumor ROI
        if keep_segs:
            vol1, vol2, seg1 = get_volumes(new_image_vol)
        else:
            vol1, vol2 = get_volumes(new_image_vol)
        if vol1 is not None:
            vol1 = vol1[None, ::]
            vol2 = vol2[None, ::]
            output[str.lower(row.Split)]["x"] = np.vstack([output[str.lower(row.Split)]["x"], vol1]) if output[str.lower(row.Split)]["x"].size else vol1
            output[str.lower(row.Split)]["x2nd"] = np.vstack([output[str.lower(row.Split)]["x2nd"], vol2]) if output[str.lower(row.Split)]["x2nd"].size else vol2
            output[str.lower(row.Split)]["label"].append(pat_clinical.EventBin.item())
            output[str.lower(row.Split)]["e"].append(pat_clinical.vital_status.item())
            output[str.lower(row.Split)]["t"].append(pat_clinical.vital_days_after_surgery.item())
            output[str.lower(row.Split)]["name"].append(pat_clinical.patient_id.item())
            output[str.lower(row.Split)]["class_365.0"].append(pat_clinical["class_365.0"].item())
            output[str.lower(row.Split)]["class_730.0"].append(pat_clinical["class_730.0"].item())
            output[str.lower(row.Split)]["class_912.5"].append(pat_clinical["class_912.5"].item())
            output[str.lower(row.Split)]["class_1277.5"].append(pat_clinical["class_1277.5"].item())
            if keep_segs:
                seg1 = seg1[None, ::]
                output[str.lower(row.Split)]["seg"] = np.vstack([output[str.lower(row.Split)]["seg"], seg1]) if output[str.lower(row.Split)]["seg"].size else seg1

# Load the rest of KITS not included in the original groupings
if len(extra_files) > 0:
    if extra_type == "Folder":
        directories = [d for d in os.listdir(extra_files) if os.path.isdir(os.path.join(extra_files, d))]
        for direc in tqdm(directories, desc="Extra Files"):
            if direc in clinical.patient_id.values:
                pat_clinical = clinical[clinical['patient_id'] == direc]
                image = ReadImage(extra_files + "/" + direc + "/imaging.nii.gz")
                new_image, new_image_vol = normalize_image(image)
                if keep_segs:
                    vol1, vol2, seg1 = get_volumes(new_image_vol)
                else:
                    vol1, vol2 = get_volumes(new_image_vol)
                if vol1 is not None:
                    vol1 = vol1[None, ::]
                    vol2 = vol2[None, ::]
                    output["exval"]["x"] = np.vstack([output["exval"]["x"], vol1]) if output["exval"]["x"].size else vol1
                    output["exval"]["x2nd"] = np.vstack([output["exval"]["x2nd"], vol2]) if output["exval"]["x2nd"].size else vol2
                    output["exval"]["label"].append(pat_clinical.EventBin.item())
                    output["exval"]["e"].append(pat_clinical.vital_status.item())
                    output["exval"]["t"].append(pat_clinical.vital_days_after_surgery.item())
                    output["exval"]["name"].append(pat_clinical.patient_id.item())
                    output["exval"]["class_365.0"].append(pat_clinical["class_365.0"].item())
                    output["exval"]["class_730.0"].append(pat_clinical["class_730.0"].item())
                    output["exval"]["class_912.5"].append(pat_clinical["class_912.5"].item())
                    output["exval"]["class_1277.5"].append(pat_clinical["class_1277.5"].item())
                    bla = 1
                    if keep_segs:
                        seg1 = seg1[None, ::]
                        output["exval"]["seg"] = np.vstack([output["exval"]["seg"], seg1]) if \
                        output["exval"]["seg"].size else seg1

hf = h5py.File(savename, 'w')
for phase in output.keys():
    group = hf.create_group(phase)
    for key in output[phase]:
        group.create_dataset(key, data=output[phase][key])
hf.close()


