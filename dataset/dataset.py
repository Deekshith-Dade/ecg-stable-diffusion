import os
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
import torch.nn.functional as F
import random




class DataLoaderError(Exception):
    pass

class ECG_KCL_Datasetloader(Dataset):
    def __init__(self, baseDir='', ecgs=[], kclVals=[], low_threshold=4.0, high_threshold=5.0, rhythmType='Rhythm',
                 allowMismatchTime=True, mismatchFix='Pad', randomCrop=False,
                 cropSize=2500, expectedTime=5000):
        self.baseDir = baseDir
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.rhythmType = rhythmType
        self.ecgs = ecgs
        self.kclVals = kclVals
        self.expectedTime = expectedTime,
        self.allowMismatchTime = allowMismatchTime
        self.mismatchFix = mismatchFix
        self.cropSize = cropSize
        self.randomCrop = randomCrop
        self.use_latents = False
        if self.randomCrop:
            self.expectedTime = self.cropSize
    
    def __getitem__(self, item):
        ecgName = self.ecgs[item].replace('.xml', f'_{self.rhythmType}.npy')
        ecgPath = os.path.join(self.baseDir, ecgName)
        ecgData = np.load(ecgPath)
        
        kclVal = torch.tensor(self.kclVals[item])
        ecgs = torch.tensor(ecgData).unsqueeze(0).float()
        # temp_ecgs = ecgs.clone()
        # ecgs[0, 0:2] = temp_ecgs[0, 2:4]
        # ecgs[0, 2:4] = temp_ecgs[0, 0:2]
        
        if self.randomCrop:
            startIx = 0
            if ecgs.shape[-1]-self.cropSize > 0:
                startIx = torch.randint(ecgs.shape[-1] - self.cropSize, (1,))
            ecgs = ecgs[...,startIx:startIx+self.cropSize]
        
        if ecgs.shape[-1] != self.expectedTime:
            if self.allowMismatchTime:
                if self.mismatchFix == 'Pad':
                    ecgs = F.pad(ecgs, (0, self.expectedTime-ecgs.shape[-1]))
                if self.mismatchFix == 'Repeat':
                    timeDiff = self.expectedTime - ecgs.shape[-1]
                    ecgs = torch.cat((ecgs, ecgs[...,0:timeDiff]))
            else:
                raise DataLoaderError('You are not allowed to have mismatching data lengths.')
        
        if torch.any(torch.isnan(ecgs)):
            print(f'Nans in the data for item {item}, {ecgPath}')
            raise DataLoaderError('Nans in data')

        item = {}
        item['image'] = ecgs
        item['y'] = 1 if kclVal <= self.high_threshold and kclVal >= self.low_threshold else 0
        item['key'] = 'kclVal'
        item['val'] = kclVal
        item['ecgPath'] = ecgPath
        cond_inputs = {}
        cond_inputs['class'] = item['y']
        if self.low_threshold <= kclVal <= self.high_threshold:
            condition_status = "Normal Signs of Hyperkalemia"
        elif self.high_threshold < kclVal <= self.high_threshold + 1:
            condition_status = "Early Stage Hyperkalemia"
        elif self.high_threshold + 1 < kclVal <= self.high_threshold + 2:
            condition_status = "Moderate Stage Hyperkalemia"
        else:
            condition_status = "Severe Stage Hyperkalemia"

        cond_inputs['text'] = f"ECG that shows {condition_status} with KCL value {kclVal:0.2f}"
        item['cond_inputs'] = cond_inputs
        return item
    
    def __len__(self):
        return len(self.ecgs)


def getKCLTrainTestDataset(dataset_config):
    randSeed = dataset_config['randSeed']
    timeCutoff = dataset_config['timeCutOff']
    lowerCutoff = dataset_config['lowerCutOff']
    data_path = dataset_config['data_path']
    dataDir = dataset_config['dataDir']
    scale_training_size = dataset_config['scale_training_size']
    kclTaskParams = dataset_config['kcl_params']
    
    assert scale_training_size <= 1.0
    
    
    np.random.seed(randSeed)
    
    # kclCohort = np.load(dataDir+'kclCohort_v1.npy',allow_pickle=True)
    # data_types = {
    #     'DeltaTime': float,   
    #     'KCLVal': float,    
    #     'ECGFile': str,     
    #     'PatId': int,       
    #     'KCLTest': str      
    # }
    kclCohort = pd.read_parquet(data_path)
    
    kclCohort = kclCohort[kclCohort['DeltaTime']<=timeCutoff]
    kclCohort = kclCohort[kclCohort['DeltaTime']>lowerCutoff]

    kclCohort = kclCohort.dropna(subset=['DeltaTime']) 
    kclCohort = kclCohort.dropna(subset=['KCLVal']) 

    ix = kclCohort.groupby('ECGFile')['DeltaTime'].idxmin()
    kclCohort = kclCohort.loc[ix]

    numECGs = len(kclCohort)
    numPatients = len(np.unique(kclCohort['PatId']))
    
    print('setting up train/val split')
    numTest = int(0.1 * numPatients)
    numTrain = numPatients - numTest
    assert (numPatients == numTrain + numTest), "Train/Test spilt incorrectly"
    RandomSeedSoAlswaysGetSameDatabseSplit = 1
    patientIds = list(np.unique(kclCohort['PatId']))
    random.Random(RandomSeedSoAlswaysGetSameDatabseSplit).shuffle(patientIds)
    
    trainPatientInds = patientIds[:numTrain]
    testPatientInds = patientIds[numTrain:numTest + numTrain]
    trainECGs = kclCohort[kclCohort['PatId'].isin(trainPatientInds)]
    testECGs = kclCohort[kclCohort['PatId'].isin(testPatientInds)]
    
    trainECGs = trainECGs[(trainECGs['KCLVal']>=kclTaskParams['lowThresh']) & (trainECGs['KCLVal']<=kclTaskParams['highThreshRestrict'])]
    testECGs = testECGs[(testECGs['KCLVal']>=kclTaskParams['lowThresh']) & (testECGs['KCLVal']<=kclTaskParams['highThreshRestrict'])]
    
    desiredTrainingAmount = int(len(trainECGs) * scale_training_size)
    print(f"{desiredTrainingAmount}: {len(trainECGs)}: {scale_training_size}")
    if desiredTrainingAmount != 'all':
        if len(trainECGs)>desiredTrainingAmount:
            trainECGs = trainECGs.sample(n=desiredTrainingAmount)
    
    dataset_regular = ECG_KCL_Datasetloader
    trainDataset = dataset_regular(
        baseDir = dataDir + 'pythonData/',
        ecgs = trainECGs['ECGFile'].tolist(),
        low_threshold= kclTaskParams['lowThresh'],
        high_threshold = kclTaskParams['highThresh'],
        kclVals=trainECGs['KCLVal'].tolist(),
        allowMismatchTime=False,
        randomCrop=True
    )
    print(f'Number of Training Examples: {len(trainDataset)}')
    
    testDataset = dataset_regular(
        baseDir = dataDir + 'pythonData/',
        ecgs = testECGs['ECGFile'].tolist(),
        low_threshold= kclTaskParams['lowThresh'],
        high_threshold = kclTaskParams['highThresh'],
        kclVals=testECGs['KCLVal'].tolist(),
        allowMismatchTime=False,
        randomCrop=True
    )

    return trainDataset, testDataset


class PreTrain_1M_Datasetloader(Dataset):
	def __init__(self,baseDir='',ecgs=[],patientIds=[], normalize =False, 
				 normMethod='0to1',rhythmType='Rhythm',allowMismatchTime=False,
				 mismatchFix='Pad',randomCrop=True,cropSize=2500,expectedTime=5000):
		self.baseDir = baseDir
		self.rhythmType = rhythmType
		self.normalize = normalize
		self.normMethod = normMethod
		self.ecgs = ecgs
		self.patientIds = patientIds
		self.expectedTime = expectedTime
		self.allowMismatchTime = allowMismatchTime
		self.mismatchFix = mismatchFix
		self.randomCrop = randomCrop
		self.cropSize = cropSize
		if self.randomCrop:
			self.expectedTime = self.cropSize

	def __getitem__(self,item):
		ecgName = self.ecgs[item].replace('.xml',f'_{self.rhythmType}.npy')
		ecgPath = os.path.join(self.baseDir,ecgName)
		ecgData = np.load(ecgPath)

		ecgs = torch.tensor(ecgData).float() #unsqueeze it to give it one channel\

		if self.randomCrop:
			startIx = 0
			if ecgs.shape[-1]-self.cropSize > 0:
				startIx = torch.randint(ecgs.shape[-1]-self.cropSize,(1,))
			ecgs = ecgs[...,startIx:startIx+self.cropSize]

		if ecgs.shape[-1] != self.expectedTime:
			if self.allowMismatchTime:
				if self.mismatchFix == 'Pad':
					ecgs=F.pad(ecgs,(0,self.expectedTime-ecgs.shape[-1]))
				if self.mismatchFix == 'Repeat':
					timeDiff = self.expectedTime - ecgs.shape[-1]
					ecgs=torch.cat((ecgs,ecgs[...,0:timeDiff]))

			else:
				raise DataLoaderError('You are not allowed to have mismatching data lengths.')

		if self.normalize:
			if self.normMethod == '0to1':
				if not torch.allclose(ecgs,torch.zeros_like(ecgs)):
					ecgs = ecgs - torch.min(ecgs)
					ecgs = ecgs / torch.max(ecgs)
				else:
					print(f'All zero data for item {item}, {ecgPath}')
			
		if torch.any(torch.isnan(ecgs)):
			print(f'Nans in the data for item {item}, {ecgPath}')
			raise DataLoaderError('Nans in data')
		batch = dict(
            image = ecgs,
            patientId = self.patientIds[item]
        )

		return batch

	def __len__(self):
		return len(self.ecgs)


def get_datasets(scale_training_size=1.0, dataset_type="1M_dataset"):
    
    if dataset_type == "1M_dataset":
        Dataset = PreTrain_1M_Datasetloader
    elif dataset_type == "KCL":
        Dataset = ECG_KCL_Datasetloader
        
    
    dataDir = '/uu/sci.utah.edu/projects/ClinicalECGs/AllClinicalECGs/'
    print('finding patients')
    df = pd.read_csv('/uu/sci.utah.edu/projects/ClinicalECGs/DeekshithMLECG/ecg_latent_diff/data/ecgs_patients_mod.csv')
    # df.to_csv('ecg_files_df.csv', index=False)
    print(f"Number of ECGs: {len(df)}")
    
    # Get unique patient IDs
    unique_patients = df['PatId'].unique()
    np.random.shuffle(unique_patients)  # Shuffle for random split
    
    # Split patients 90/10
    split_idx = int(0.9 * len(unique_patients))
    train_patients = unique_patients[:split_idx]
    # If scale_training_size is less than 1.0, adjust the training set size
    if scale_training_size < 1.0:
        train_patients = train_patients[:int(len(train_patients) * scale_training_size)]
    val_patients = unique_patients[split_idx:]
    
    print(f"Total patients: {len(unique_patients)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    
    # Split dataframe based on patient IDs
    train_df = df[df['PatId'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['PatId'].isin(val_patients)].reset_index(drop=True)
    
    print(f"Train ECGs: {len(train_df)}")
    print(f"Validation ECGs: {len(val_df)}")
    
    # Create datasets
    train_dataset = PreTrain_1M_Datasetloader(
        baseDir=dataDir + 'pythonData/',
        ecgs=train_df['ECGFile'].tolist(),
        patientIds=train_df['PatId'].tolist(),
        normalize=False,
        allowMismatchTime=False,
        randomCrop=True,
    )
    
    val_dataset = PreTrain_1M_Datasetloader(
        baseDir=dataDir + 'pythonData/',
        ecgs=val_df['ECGFile'].tolist(),
        patientIds=val_df['PatId'].tolist(),
        normalize=True,
        normMethod='0to1',
        randomCrop=False,  # Usually don't crop validation data
        cropSize=2500
    )
    
    return train_dataset, val_dataset