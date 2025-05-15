# Renal Cortex and Medulla Segmentation in Arterial Phase CT  

This project focuses on the separate segmentation of renal cortex and medulla in arterial phase CT scans using deep learning. The model is trained to accurately segment these structures along with renal masses (cysts and tumors).  

## Dataset  
The model is trained and evaluated on the publicly available dataset from the [KiTS21 challenge](https://kits21.kits-challenge.org/). The ground truth segmentation masks are provided in `.nrrd` file format.  

## Segmentation Classes  
The network predicts the following classes:  
- **Class 0:** Background  
- **Class 1:** Renal cortex  
- **Class 2:** Renal medulla  
- **Class 3:** Renal masses (cysts and tumors)  

## Setup  

### 1. Conda Environment  
A `requirements.txt` file is provided for setting up the environment.  

#### Create and activate the conda environment:  
```bash  
conda create --name pytorch1.10 --file requirements.txt  
conda activate pytorch1.10  
```  

#### Verify the environment installation:  
```bash  
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \ 
           import monai; print(f'MONAI version: {monai.__version__}'); \ 
           print(f'CUDA available: {torch.cuda.is_available()}'); \  
           print(f'GPU devices: {torch.cuda.device_count()}')"  
```  
This should display:  
- The installed PyTorch version  
- The install MONAI version
- Whether CUDA is available (should be `True`)  
- The number of available GPUs  

### 2. CUDA Configuration (Optional)  
Specify which GPUs to use (comma-separated, no spaces):  
```bash  
export CUDA_VISIBLE_DEVICES=deviceID1,deviceID2,...  
```  
If not set, the program will use all available GPUs.  

### 3. Running the Program  
Execute the main script with the configuration file:  
```bash  
python main.py -c=./config.yml  
```  

The program includes comprehensive logging functionality. The following files will be generated in the directory `./output/logdir`, with `logdir` specified in `config.yml`:  
1. **training_logs.log** - Contains detailed information about the training process i.e. loss values, dice score per epoch 
2. **config.yml** - A copy of the configuration file used for the experiment  
3. **models/model.tar** - The best performing model weights saved during training  

## Configuration  
The `config.yml` file contains all training and evaluation parameters, including:  
- Log file directory path  
- Model hyperparameters  
- Training settings  
- Validation parameters  

Adjust it as needed for your setup.  

## Link to MS Teams meeting

https://teams.microsoft.com/l/meetingrecap?driveId=b%21Jlq4QIbqcE2a64a0A1gQaedfa-4hRAFAmOt1Z7nmfiFUSd2wqdQISYGN4cuIHSDP&driveItemId=01ULXUSD6QP6LWC7W3UREY42JNTN6U5C4B&sitePath=https%3A%2F%2Fuzleuven-my.sharepoint.com%2F%3Av%3A%2Fg%2Fpersonal%2Fkonstantinos_koukoutegos_uzleuven_be%2FEdB_l2F-26RJjmktm31Oi4EBz9-1XLdgkng_b2CsIMi4tg&fileUrl=https%3A%2F%2Fuzleuven-my.sharepoint.com%2Fpersonal%2Fkonstantinos_koukoutegos_uzleuven_be%2FDocuments%2FOpnamen%2FUpdate%2520on%2520the%2520renal%2520cortex%2520and%2520medulla%2520segmentation-20250515_160550-Meeting%2520Recording.mp4%3Fweb%3D1&iCalUid=040000008200E00074C5B7101A82E00800000000107AC9E2AEC4DB010000000000000000100000004A47C4836BE0A643AD29DE661CBEFA27&threadId=19%3Ameeting_MGQwZDIwNTctNGQ2ZS00ZTRkLTg1NGEtMDJkZDAwYmUyMGRk%40thread.v2&organizerId=d60ca9b3-4aad-4767-b8f5-e11b3e251407&tenantId=a8da9b9a-de5a-4522-8935-b6c421e93e7c&callId=f7d0ee84-57e2-407c-b4da-81c56ada7af5&threadType=Meeting&meetingType=Scheduled&subType=RecapSharingLink_RecapCore
