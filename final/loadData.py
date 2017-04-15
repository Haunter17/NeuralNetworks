import pandas as pd
import numpy as np
# file configurations
filePath = 'kdd_small.csv'
features = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', \
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', \
'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', \
'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', \
'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', \
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', \
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', \
'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', \
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', \
'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']
df = pd.read_csv(filePath, names = features)
df = df.sample(frac = 1).reset_index(drop = True) # shuffle rows
continuous_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', \
'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', \
'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', \
'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', \
'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', \
'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', \
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', \
'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
category_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', \
'is_host_login', 'is_guest_login']
# hash categorical features into floats
for category in category_features:
	df[category] = df[category].astype('category')
	df[category] = getattr(df, category).cat.codes
# hash labels
df['label'] = df['label'].astype('category')
df['label'] = df.label.cat.codes
# partition data
df['type'] = ''
df.loc[0: int(0.75 * len(df)) - 1, 'type'] = 'train'
df.loc[int(0.75 * len(df)): , 'type'] = 'test'
# # convert data into matrix
X_train = df[df.type == 'train'][[col for col in df.columns if col not in ['label', 'type']]]
y_train = df[df.type == 'train'].label

X_test = df[df.type == 'test'][[col for col in df.columns if col not in ['label', 'type']]]
y_test = df[df.type == 'test'].label
