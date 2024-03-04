import numpy as np
from rdkit import Chem
import pandas as pd
import pickle
import os
from config import fpFunc_dict
import argparse
from sklearn.impute import SimpleImputer

class FeaturesGeneration:
    def __init__(self, file_name):
        self.fingerprints = []
        self.file_name = file_name
    def get_fingerprints(self, df, model, fp_name, split, numpy_folder):
        smiles_list = df['SMILES_stand'].to_list()
        not_found = []
        for smi in smiles_list:
            try: 
                m = Chem.MolFromSmiles(smi)
                can_smi = Chem.MolToSmiles(m, True)
                fp = fpFunc_dict[fp_name](m)
                bit_array = np.asarray(fp)
                self.fingerprints.append(bit_array)
            except:
                not_found.append(smi)
                if fp_name == 'tpatf':
                    add = [np.nan for i in range(self.fingerprints[0].shape[1])]
                elif fp_name == 'rdkDes':
                    add = [np.nan for i in range(len(self.fingerprints[0]))]
                else:
                    add = [np.nan for i in range(len(self.fingerprints[0]))]
                tpatf_arr = np.array(add, dtype=np.float32)
                self.fingerprints.append(tpatf_arr)
                pass
        print('not_found', len(not_found))
        if fp_name == 'rdkDes':
            X = np.array(self.fingerprints)
            ndf = pd.DataFrame.from_records(X)
            ndf.isnull().sum().sum()
            r, _ = np.where(df.isna())
            ndf.isnull().sum().sum()
            for col in ndf.columns:
                ndf[col].fillna(ndf[col].mean(), inplace=True)
            ndf.isnull().sum().sum()
            X = ndf.iloc[:,0:].values
            fp_array = ( np.asarray((X), dtype=object) )
            X = X.astype(np.float32)
            X = np.nan_to_num(X)
            rdkDes_scaler = pickle.load(open('../scalers/'+model+'-rdkDes_scaler.pkl', 'rb'))
            X = rdkDes_scaler.transform(X)
        else:
            fp_array = ( np.asarray((self.fingerprints), dtype=object) )
            X = np.vstack(fp_array).astype(np.float32)
            imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
            imp_median.fit(X)  
            X = imp_median.transform(X)
        self.fingerprints = []
        print('Final shape: ', X.shape, 'Not found: ', not_found)
        np.save(numpy_folder+'/'+fp_name+'-'+self.file_name+'.npy', X)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Features Extraction")
    parser.add_argument('--csvfile', action='store', dest='csvfile', required=True, \
                        help='csvfile needed with SMILES_stand')
    parser.add_argument('--model', action='store', dest = 'model', required=True,\
                        help='model type needed')
    parser.add_argument('--split-type', action='store', dest = 'split-type', required=True,\
                        help='split-type tr,va,te,ext needed')
    parser.add_argument('--numpy_folder', action='store', dest = 'numpy_folder', required=True,\
                        help='folder for numpy files is needed')    
    parser.add_argument('--ft', action='store', dest= 'ft', required=True,\
            help='provide feature type')
    args = vars(parser.parse_args())
    csv_file = args['csvfile']
    model = args['model']
    split = args['split-type']
    features = args['ft']
    numpy_folder = args['numpy_folder']
    
    """
    fp_list = ['ecfp0', 'ecfp2', 'ecfp4', 'ecfp6', 'fcfp2', 'fcfp4', 'fcfp6', 'lecfp4', 'lecfp6',\
            'lfcfp4', 'lfcfp6', 'maccs', 'hashap', 'hashtt', 'avalon', 'laval', 'rdk5', 'rdk6',\
            'rdk7', 'tpatf']
    """
    
    df = pd.read_csv(csv_file)
    file_name,_ = os.path.splitext(os.path.basename(csv_file)) 
    fg = FeaturesGeneration(file_name)
    fg.get_fingerprints(df, model, features, split, numpy_folder)
