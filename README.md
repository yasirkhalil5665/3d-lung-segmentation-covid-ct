# COVID-19 CT Lung Segmentation with MONAI

A 3D U-Net, built with [MONAI](https://monai.io) and PyTorch, that segments the **left and right lung** on chest CT scans from COVID-19 patients. Predictions are saved as NIfTI files and can be inspected in [3D Slicer](https://www.slicer.org).

> Learning project. Not a medical device and not for clinical use.

## What it does

1. Pairs each CT scan with its lung mask (matched by file name)
2. Preprocesses: RAS orientation, resampling to a fixed voxel size, CT windowing (-1000 to 400 HU)
3. Trains a 3D U-Net on random 3D patches (Dice + cross-entropy loss)
4. Validates on held-out scans with sliding-window inference and the Dice metric
5. Plots training curves and compares predictions with the true masks
6. Saves predictions as `.nii.gz` for viewing in 3D Slicer

Output classes: `0` background, `1` left lung, `2` right lung.

## Dataset

[COVID-19 CT Lung and Infection Segmentation Dataset](https://zenodo.org/records/3757476) (Zenodo, 20 labeled CT cases from Coronacases Initiative and Radiopaedia). Check the Zenodo page for the license and citation requirements.

Related paper: *Towards Efficient COVID-19 CT Annotation: A Benchmark for Lung and Infection Segmentation* ([arXiv:2004.12537](https://arxiv.org/abs/2004.12537)).

The data is **not included** in this repository. Download it as shown below.

## Setup

Python 3.11 or 3.12 is recommended (some MONAI-related packages do not yet build on the newest Python versions).

```bash
git clone <your-repo-url>
cd <your-repo>

python3.11 -m venv monai-env311
source monai-env311/bin/activate
pip install -r requirements.txt
```

No NVIDIA GPU? Install the smaller CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Get the data

```bash
mkdir -p data/Training_CT data/Lung_Mask

wget -O ct.zip   "https://zenodo.org/records/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1"
wget -O mask.zip "https://zenodo.org/records/3757476/files/Lung_Mask.zip?download=1"

unzip ct.zip   -d data/Training_CT
unzip mask.zip -d data/Lung_Mask
```

Expected layout:

```
data/
  Training_CT/   # CT scans, e.g. coronacases_001.nii.gz
  Lung_Mask/     # masks with identical file names
```

## Run

```bash
jupyter lab
```

Open `all.ipynb` and run the cells top to bottom. Settings switch automatically:

| | CPU | GPU |
|---|---|---|
| Epochs | 10 | 100 |
| Patch size | 64³ | 96³ |
| Voxel size | 3 mm | 2 mm |
| U-Net channels | 8-16-32-64 | 16-32-64-128-256 |

A short CPU run only checks that the pipeline works. For good results, train on a GPU (for example the free Colab T4).

## Viewing results in 3D Slicer

1. Load the CT (for example `radiopaedia_36_86526_0.nii.gz`).
2. Load the prediction from `predictions/` and the true mask from `data/Lung_Mask/` as **Segmentation**.
3. In the **Segmentations** module, click **Show 3D**.
4. Toggle the eye icons to compare the prediction with the true mask.

The validation scans are the last four files in alphabetical order (the `radiopaedia_*` cases).

## Results

Add your numbers here after training:

| Setup | Epochs | Validation Dice |
|---|---|---|
| CPU test run | 10 | _fill in_ |
| GPU run | 100 | _fill in_ |

## Project structure

```
.
├── 3D slicer/           # Visualization examples
├── data/                # Dataset (Training_CT, Lung_Mask)
├── models/               # Saved model weights (.pth)
├── predictions/         # NIfTI prediction files
├── all.ipynb            # full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Possible next steps

- Segment the COVID **infection** regions (`Infection_Mask.zip` from the same Zenodo record)
- Shuffle the train/validation split so both data sources appear in each
- Try other MONAI networks (SegResNet, Swin UNETR)
- Use MONAI Label with 3D Slicer for AI-assisted annotation

## Acknowledgements

Built with [MONAI](https://monai.io), [PyTorch](https://pytorch.org) and [3D Slicer](https://www.slicer.org). Dataset by Ma et al. (see the Zenodo record above).
