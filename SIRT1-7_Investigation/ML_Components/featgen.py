#! /usr/bin/env python

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdmolfiles
import itertools
from collections import Counter


class Generate_Features():
    def __init__(self, user_feat, protdcal, goa):
        self.user_feat = user_feat
        self.protdcal = protdcal
        self.goa = goa
    
    # GENERATE FEATURE FOR SEQUENCE
    def FeatureGen (self, sequence, protdcal):
        # METHOD FOR GENERATING FEATURES USING ONE SEQUENCE AT A TIME
        # FIRST: GENERATE PROTDCAL VALUES
        slist = list(sequence)   # first split sequence up into list
        # Go through sequence to get protdcal value
        pd = []
        for i in slist:
            pd.append(protdcal.loc[i].tolist())
        values = list(map(lambda *x: sum(x), *pd))   # add up values
        headers =  protdcal.columns.tolist()   # include headers
        
        
        # SECOND: GENERATE ONE-HOT ENCODING
        aa = ['K', 'R', 'H', 'A', 'I', 'L', 'M', 'V', 'F', 'W', 'Y', 'N', 'C', 'Q', 'S', 'T', 'D', 'E', 'G', 'P']   # possible amino acids
        # Make headers and one-hot encoding for each letter
        for i in aa:
            j = 0
            while j < len(sequence):
                headers.append('ONE-HOT_' + str(j) + '-' + i)  # make header
                if sequence[j] == i:
                    values.append(1)
                else:
                    values.append(0)
                j+=1
        
        
        # THIRD: GENERATE MACCS KEYS
        # Generate maccs keys
        mol = (rdmolfiles.MolFromFASTA(sequence))
        fp = (MACCSkeys.GenMACCSKeys(mol))
        maccs = fp.ToBitString()
        binary = list(maccs)   # split up into list
        values.extend(binary)   # add list onto resulting values
        # Generate headers for maccs keys
        mt = list(itertools.chain(range(len(binary))))
        mt = [str(s) + '_maccs' for s in mt]
        headers.extend(mt)   # append header values
        
        return values, headers
    
    def feature_generate(self):
        ## MOVED THIS UP HERE
        # Remove duplicates
        feat_dup_len = (len(self.user_feat))
        self.user_feat = self.user_feat.drop_duplicates(subset='SITE_+/-7_AA')
        feat_dup_len = feat_dup_len - (len(self.user_feat))

        # Format user defined features for FeatureGen call
        temp_annotation = self.user_feat['SITE_LOC'].astype(str)
        self.user_feat['uid_pos'] = self.user_feat['ACC_ID'] + "_" + temp_annotation
        self.user_feat['uid_pos'] = self.user_feat['uid_pos'] + '_c' + self.user_feat.groupby('uid_pos').cumcount().astype(str) ## ADDED IN TO ADDRESS SAME UID_POS BUT DIFF SEQ ISSUE
        self.user_feat = self.user_feat.set_index('uid_pos', drop=True)

        # Pull sequences and make X -> A for feature generation (A is least offensive AA)
        sequences = self.user_feat['SITE_+/-7_AA'].str.replace('X', 'A')

        # Pull user defined features' uniprot IDs from GO table - generate associated GO annotations
        feat_uids = pd.DataFrame(self.user_feat['ACC_ID'])
        feat_uids = feat_uids.rename(columns={'ACC_ID':'uid'})
        feat_uids = feat_uids.drop_duplicates().reset_index(drop=True)
        goa_uids = pd.merge(feat_uids, self.goa, on='uid')
        goa_counts = pd.DataFrame(goa_uids['go_term'].value_counts(), columns=['count'])
        goa_counts = goa_counts.reset_index(drop=False).rename(columns={'index':'go_term', 'count':'go_count'})
        goa_unique = goa_uids.drop_duplicates(subset='go_term')
        mapped_goa = pd.merge(goa_counts, goa_unique, on='go_term').drop(columns=['uid', 'gene', 'go_association', 'go_ref', 'go_b_p_f'])   # finalised df with counts to be displayed to user

        # Now get into generating our features!

        # Create df for results to go into
        v, h = self.FeatureGen(sequences[0], self.protdcal)
        features = pd.DataFrame(columns=h)
        features.loc[len(features)] = v

        i = 1

        # Go through rest of sequences to generate feature set
        while i < len(sequences):
            ts = sequences[i]
            value, header = self.FeatureGen(ts, self.protdcal)
            print(len(features))
            features.loc[len(features)] = value
            i+=1
            if i % 500 == 0:
                print(i, 'of', len(sequences), 'completed')

        # Make the index the same as our initial dataframe
        feat_x = features.set_index(self.user_feat.index)

        # Isolate the methylated condition from the sequences as our y value
        feat_y = self.user_feat['EXPERIMENTALLY_ACTIVE']


        if 'SECONDARY_ML_SCORE' not in self.user_feat.columns or (self.user_feat['SECONDARY_ML_SCORE'] == 0).all():
            print("Cannot proceed with Meta Learning as the SECONDARY_ML_SCORE column in the feature dataset doesn't contain any scores.\nTo proceed, populate the SECONDARY_ML_SCORE column with numeric scores from another computational method.")
            ms_full = None
        else:
            print("SECONDARY_ML_SCORE successfully uploaded. Proceed with Meta Learning!")
            feat_x['SECONDARY_ML_SCORE'] = self.user_feat['SECONDARY_ML_SCORE']
            #ms_full = user_feat[['SECONDARY_ML_SCORE', 'ACC_ID', 'SITE_LOC']]

        print ("Feature Generation has finished! \nThe chosen dataset has " + str(Counter(feat_y)[1]) + " positives and " + str(Counter(feat_y)[0]) + " negatives.\n"+str(feat_dup_len)+" duplicated features removed.\nYou may proceed with Model Fitting by clicking the next tab.")

        return feat_y, feat_x, self.user_feat, mapped_goa

