import pandas as pd
import numpy as np
# file configurations
filePath = 'kdd_10.csv'
portion = 0.5
train_portion = 0.5
# 'num_root', 'num_file_creations', 'num_shells' are with mixed types
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
df = pd.read_csv(filePath, names = features, low_memory = False)

# convert a few mixed continuous features into integers
for feature in ['num_root', 'num_file_creations', 'num_shells']:
	df[feature] = df[feature].astype('category')
	df[feature] = getattr(df, feature).cat.codes
	df[feature] = df[feature].astype('int64')
# hash categorical features into integers
for category in category_features:
	df[category] = df[category].astype('category')
	df[category] = getattr(df, category).cat.codes
# hash labels
df['label'] = df['label'].astype('category')
df['label'] = df.label.cat.codes # 11 stands for normal

# partition data
df = df.sample(frac = 1).reset_index(drop = True) # shuffle rows
df['type'] = ''
df = df.loc[0:int(portion * len(df))]
df.loc[0: int(train_portion * len(df)) - 1, 'type'] = 'train'
df.loc[int(train_portion * len(df)): , 'type'] = 'test'
X_train = df[df.type == 'train'][[col for col in df.columns if col not in ['label', 'type']]]
y_train = df[df.type == 'train']['label']
X_test = df[df.type == 'test'][[col for col in df.columns if col not in ['label', 'type']]]
y_test = df[df.type == 'test']['label']
# convert into array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
