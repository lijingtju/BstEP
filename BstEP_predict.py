import pandas as pd
from smile_standardization import StandardSmiles
import numpy as np
import pickle
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import Chem
import os
import argparse

def standardize(input_path, file_name):
    df = pd.read_csv(input_path + file_name)
    print("shape:", df.shape)
    smiles = df['SMILES']
    smiles_standard = []
    for i in range(len(smiles)):
        sd = StandardSmiles()
        stand_smi = sd.preprocess_smi(smiles[i])
        smiles_standard.append(stand_smi)
    df.insert(1, 'SMILES_stand', smiles_standard)
    df['SMILES_stand'].replace('', np.nan, inplace=True)
    df.dropna(subset=['SMILES_stand'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(input_path + file_name[:-4] + "_stand.csv", index=False)
    print(input_path + file_name[:-4])
    print("shape:", df.shape)
    return df

def get_rdk_features(data_path, file_name, save_folder):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    fpdict ={}
    fpdict['rdkDes'] = lambda m: calc.CalcDescriptors(m)
    print(fpdict)
    def CalculateFP(fp_name, smiles):
        m = Chem.MolFromSmiles(smiles)
        return fpdict[fp_name](m)
    trg = []
    des = []
    not_found = []
    df = pd.read_csv(data_path+file_name)
    clean_smi = df['SMILES_stand'].tolist()
    rdkit_des = []
    for i in range(len(clean_smi)):
        fp = CalculateFP('rdkDes', clean_smi[i])
        fp = np.asarray(fp)
        fp = fp.reshape(1,200)
        rdkit_des.append(fp)
    X = np.array(rdkit_des)
    X = X.reshape(len(rdkit_des),200)
    ndf = pd.DataFrame.from_records(X)
    ndf.isnull().sum().sum()
    r, _ = np.where(df.isna())
    ndf.isnull().sum().sum()
    for col in ndf.columns:
        ndf[col].fillna(ndf[col].mean(), inplace=True)
    ndf.isnull().sum().sum()
    X = ndf.iloc[:, 0:].values
    X = np.vstack(X).astype(np.float32)
    X = np.nan_to_num(X)
    fp_array = ( np.asarray((X), dtype=object) )
    X = np.delete(fp_array, not_found, axis=0)
    X = X.astype(np.float32) 
    print(fp_array.shape)
    out = os.path.splitext(os.path.basename(file_name))[0]
    trg.append(out)
    np.save(save_folder+'rdkDes-'+out+'.npy', np.asarray((X), dtype=np.float32))

def get_df_prob(viru_name, finger_list,finger_model_list):
    prob_list = []
    for des, cla in zip(finger_list,finger_model_list):
        models_name = des+"-"+viru_name+"_"+cla+".pkl"
        with open(model_path+models_name, 'rb') as f:
            model = pickle.load(f)
        data_name = des+"-"+file_name[:-4]+"_stand.npy"
        x = np.load(path_fea+data_name)
        y_proba = model.predict_proba(x)[:, 1]
        prob_list.append(y_proba)
    arr_prob = np.array(prob_list).reshape(-1, x.shape[0])
    df_prob = (pd.DataFrame(arr_prob)).T
    return df_prob

def get_best_weights(weights_combinations, finger_prob, df_tpatf_prob, df_rdkDes_prob, threshold):
    for weights in weights_combinations:
        w1,w2,w3=weights[0],weights[1],weights[2]
        weighted_prob_list = []
        for p1, p2, p3 in zip(finger_prob.tolist(), df_tpatf_prob[0].values.tolist(), df_rdkDes_prob[0].values.tolist()):
            weighted_prob = (w1 * p1 + w2 * p2 + w3 * p3) / sum(weights)
            weighted_prob_list.append(weighted_prob)
        y_pred = ["active" if prob > threshold else "in-active" for prob in weighted_prob_list]
    return y_pred, weighted_prob_list

def get_test_results(viru_name, best_weights, finger_list,finger_model_list, tpatf_list, tpatf_model_list, rdkDes_list, rdkDes_model_list, threshold):
    df_finger_prob = get_df_prob(viru_name, finger_list,finger_model_list)
    finger_prob = df_finger_prob.sum(axis=1)/df_finger_prob.shape[1]
    df_tpatf_prob = get_df_prob(viru_name, tpatf_list, tpatf_model_list)
    df_rdkDes_prob = get_df_prob(viru_name, rdkDes_list, rdkDes_model_list)
    y_pred, y_prob = get_best_weights([best_weights], finger_prob, df_tpatf_prob, df_rdkDes_prob, threshold)
    return y_pred, y_prob


def main(data_path, file_name, output_path, outputfile):
    standardize(data_path, file_name)
    file_name_stand = file_name[:-4]+'_stand.csv'
    # ''' 
    # feature generation.
    # '''
    get_rdk_features(data_path, file_name_stand, path_fea)
    print('rdkDes end.......')
    des_list = ["ecfp4", "hashap", "lfcfp6", "maccs","lfcfp4", "fcfp4", "rdk5"]
    for des in des_list:
        os.system('python3 ./feature_generation_3model.py --csvfile '+data_path+file_name_stand+' --model EV71 --split-type te --ft '+des+' --numpy_folder '+ path_fea)
    print('finger descriptors end.......')
    fw = open('./tpatf_temp.sh', 'w')
    fw.write('python3 ./feature_generation_3model.py --csvfile '+data_path+file_name_stand+' --model EV71 --split-type te --ft tpatf --numpy_folder '+path_fea)
    fw.close()
    os.system("bash ./tpatf_temp.sh")
    print('tpatf end.......')
    # # #  this is for EV-A71
    data = pd.read_csv("./data/"+file_name_stand)
    EV71_finger_list = ["ecfp4", "hashap", "lfcfp6"]
    EV71_finger_model_list = ["ADA", "KNB", "DT"]
    EV71_tpatf_list = ["tpatf"]
    EV71_tpatf_model_list = ["DT"]
    EV71_rdkDes_list = ["rdkDes"]
    EV71_rdkDes_model_list = ["ADA"]
    EV71_best_weights = (2,1,4)
    EV71_threshold = 0.5
    EV71_pred, EV71_prob = get_test_results("EV71", EV71_best_weights, EV71_finger_list,EV71_finger_model_list, EV71_tpatf_list, EV71_tpatf_model_list, EV71_rdkDes_list, EV71_rdkDes_model_list, EV71_threshold)
    data.insert(0, "EV71_prob", EV71_prob)
    data.insert(0, "EV71_pred", EV71_pred)
    # # #  this is for SARS-CoV-2
    SARS_finger_list = ["maccs", "maccs", "hashap"]
    SARS_finger_model_list = ["DT", "RF", "SVC"]
    SARS_tpatf_list = ["tpatf"]
    SARS_tpatf_model_list = ["ADA"]
    SARS_rdkDes_list = ["rdkDes"]
    SARS_rdkDes_model_list = ["DT"]
    SARS_best_weights = (2,4,3)
    SARS_threshold = 0.5
    SARSC_CPE_pred, SARS_CPE_prob = get_test_results("SARS_CPE", SARS_best_weights, SARS_finger_list,SARS_finger_model_list, SARS_tpatf_list, SARS_tpatf_model_list, SARS_rdkDes_list, SARS_rdkDes_model_list, SARS_threshold)
    data.insert(0, "SARS_CPE_prob", SARS_CPE_prob)
    data.insert(0, "SARS_CPE_pred", SARSC_CPE_pred)
    # # #  this is for EV-A71
    H1N1_finger_list = ["lfcfp4", "fcfp4", "rdk5"]
    H1N1_finger_model_list = ["DT", "DT", "DT"]
    H1N1_tpatf_list = ["tpatf"]
    H1N1_tpatf_model_list = ["DT"]
    H1N1_rdkDes_list = ["rdkDes"]
    H1N1_rdkDes_model_list = ["ETAs"]
    H1N1_best_weights = (1,1,1)
    H1N1_threshold = 0.5
    H1N1_pred, H1N1_prob = get_test_results("H1N1", H1N1_best_weights, H1N1_finger_list,H1N1_finger_model_list, H1N1_tpatf_list, H1N1_tpatf_model_list, H1N1_rdkDes_list, H1N1_rdkDes_model_list, H1N1_threshold)
    data.insert(0, "H1N1_prob", H1N1_prob)
    data.insert(0, "H1N1_pred", H1N1_pred)
    data.insert(0,'SMILES_stand', data.pop("SMILES_stand"))
    data.insert(0,'SMILES', data.pop("SMILES"))
    data.to_csv(output_path + outputfile, index=False, index_label = 'index_label')

if __name__ == '__main__':
    path_fea = "./features/"
    model_path = "./model/"
    parser = argparse.ArgumentParser(description="screening the Broad-spectrum antiviral drugs")
    parser.add_argument('--csvfile', action='store', dest='csvfile', required=True, \
                        help='csvfile needed with blood laboratory test results')
    parser.add_argument('--outputfile', action='store', dest='outputfile', required=True, \
                        help='outputfile with save path and save name')
    args = vars(parser.parse_args())
    csv_file = args['csvfile']
    output_file = args['outputfile']
    data_path = os.path.dirname(csv_file)+"/"
    file_name = os.path.basename(csv_file)
    output_path = os.path.dirname(output_file)+"/"
    outputfile = os.path.basename(output_file)
    print(data_path, file_name, output_path, outputfile)
    main(data_path, file_name, output_path, outputfile)