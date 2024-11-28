# Implementation-code
Coding Part
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install Boruta


# In[ ]:


get_ipython().system('pip install capsule')


# In[ ]:


#Importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import log_loss
sns.set_style("darkgrid")
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Bidirectional, Dropout, concatenate
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, multilabel_confusion_matrix
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc, mean_squared_error,  mean_absolute_error, jaccard_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from tensorflow.keras.layers import Dense, Conv1D,MaxPooling1D, Flatten
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from scipy.stats import zscore
from scipy import stats
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as pr_auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
#Importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import log_loss
sns.set_style("darkgrid")
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Bidirectional, Dropout, concatenate
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, multilabel_confusion_matrix
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_curve, auc, mean_squared_error,  mean_absolute_error, jaccard_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef, roc_curve, auc
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tensorflow.keras.layers import Dense, Conv1D,MaxPooling1D, Flatten
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
from sklearn.utils import shuffle
from scipy.stats import zscore
from scipy import stats
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from tensorflow.keras import layers
import capsule as cp
import tensorflow as tf
#from capsule.layers import CapsuleLayer1
#from capsule.layers import CapsuleLayer, PrimaryCapsuleLayer
#from keras_contrib.layers import Capsule
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Attention, Conv1D, GlobalMaxPooling1D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import LearningRateScheduler


# In[ ]:


# Step 2: Load and Preprocess Data
data = pd.read_csv('partcombo-00033-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv')


# In[ ]:


import numpy as np
data.replace(' ', np.nan, inplace=True)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
# Split features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# One hot encode the dependent variable
encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Label encode the independent variables
label_encoder = LabelEncoder()
for i in range(X.shape[1]):
    X[:, i] = label_encoder.fit_transform(X[:, i])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


# Apply SMOTE oversampling
#smote = SMOTE(sampling_strategy='auto', random_state=42)
#X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.8, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,train_size=0.8, test_size=0.2, random_state=42)


# In[ ]:


class_counts = data['label'].value_counts()

print(class_counts)


# In[ ]:


data.corr()


# In[ ]:


# split the dataset into training, testing, and validation datasets
#X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.decomposition import TruncatedSVD

# Initialize and fit Boruta
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', max_depth=100)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, max_iter=100)

# Apply TruncatedSVD
svd = TruncatedSVD(n_components=42)  # Adjust the number of components as needed
X_train_svd = svd.fit_transform(X_train)

# Fit Boruta on the dense array
boruta_selector.fit(X_train_svd, y_train[:, 0])

# Get the ranks and selected feature indices
selected_feature_indices = np.where(boruta_selector.support_)[0]

# Extract importance scores from the RandomForest model used in Boruta
rf.fit(X_train_svd, y_train[:, 0])  # Re-fit the RF model on the same data
importance_scores = rf.feature_importances_[selected_feature_indices]

# Get the column names of the selected features
best_features = np.array(data.columns[:-1])[selected_feature_indices]

# Plot the selected features and their importance scores
plt.figure(figsize=(10, 6))
sns.barplot(x=best_features, y=importance_scores)
plt.title("Selected Feature Importance After Boruta Feature Selection")
plt.xlabel("Selected Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=90)
plt.show()

# Drop highly correlated variables (Optional)
correlation_matrix = pd.DataFrame(X_train, columns=data.columns[:-1])[best_features].corr()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

X_train_filtered = pd.DataFrame(X_train, columns=data.columns[:-1])[best_features].drop(to_drop, axis=1)
X_val_filtered = pd.DataFrame(X_val, columns=data.columns[:-1])[best_features].drop(to_drop, axis=1)
X_test_filtered = pd.DataFrame(X_test, columns=data.columns[:-1])[best_features].drop(to_drop, axis=1)


# In[ ]:


from boruta import BorutaPy

rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', max_depth=100)
#dt = DecisionTreeClassifier(n_estimators=50, n_jobs=-1, class_weight='balanced', max_depth=10)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, max_iter=100)
#boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
#from sklearn.decomposition import PCA

# Instantiate PCA and fit on sparse data
#pca = PCA(n_components=100)  # Adjust the number of components as needed
#X_train_pca = pca.fit_transform(X_train)

from sklearn.decomposition import TruncatedSVD

# Instantiate TruncatedSVD and fit on sparse data
svd = TruncatedSVD(n_components=42)  # Adjust the number of components as needed
X_train_svd = svd.fit_transform(X_train)

# Now convert the SVD-transformed data to a dense array
X_train_dense = X_train_svd


# Now convert the PCA-transformed data to a dense array
#X_train_dense = X_train_pca.toarray()

#boruta_selector.fit(X_train, y_train[:, 0])  # Assuming you want to select features for the first class
##X_train_dense = X_train.toarray()
boruta_selector.fit(X_train_dense, y_train[:, 0])


# In[ ]:


# Plot the best features
feature_ranks = boruta_selector.ranking_

# Get the selected feature indices

selected_feature_indices = np.where(boruta_selector.support_)[0]
# Get the column names of the selected features
#best_features = np.array(data.columns[:-1])[selected_feature_indices]
best_features = np.array(data.columns)[: -1][selected_feature_indices]


# Create numerical indices for columns
column_indices = np.arange(X.shape[1])

# Plot the ranking of features
plt.figure(figsize=(10, 6))
#sns.barplot(x=feature_ranks, y=column_indices)
sns.barplot(x=feature_ranks[selected_feature_indices], y=best_features)
plt.title("Boruta Feature Ranking")
plt.xlabel("Feature Rank")
plt.ylabel("Feature Name")
plt.show()


# In[ ]:


print("Minimum selected feature index:", np.min(selected_feature_indices))
print("Maximum selected feature index:", np.max(selected_feature_indices))


# In[ ]:


print("Number of columns in dataset:", len(data.columns[:-1]))
print("Number of selected features:", len(selected_feature_indices))
print("Number of selected features:", np.sum(boruta_selector.support_))


# In[ ]:


# Plot the ranking of features with values
plt.figure(figsize=(10, 6))
barplot = sns.barplot(x=feature_ranks[selected_feature_indices], y=best_features)

# Display values on top of the bars
for index, value in enumerate(feature_ranks[selected_feature_indices]):
    barplot.text(value, index, f'{value:.2f}', ha="left", va="center")

plt.title("Boruta Feature Ranking")
plt.xlabel("Feature Rank")
plt.ylabel("Feature Name")
plt.show()


# In[ ]:


# Get the column names of the selected features
best_features = np.array(data.columns[:-1])[selected_feature_indices]

# Drop highly correlated variables
correlation_matrix = pd.DataFrame(X_train, columns=data.columns[:-1])[best_features].corr()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

X_train_filtered = pd.DataFrame(X_train, columns=data.columns[:-1])[best_features].drop(to_drop, axis=1)
X_val_filtered = pd.DataFrame(X_val, columns=data.columns[:-1])[best_features].drop(to_drop, axis=1)
X_test_filtered = pd.DataFrame(X_test, columns=data.columns[:-1])[best_features].drop(to_drop, axis=1)


# In[ ]:


import matplotlib.pyplot as plt

# Assuming 'importance_scores' is an array containing the importance score for each feature
# You need to have 'importance_scores' calculated or obtained from a model
# For example, from a RandomForest model: importance_scores = model.feature_importances_

# Filter the importance scores to match the selected (and non-correlated) features
filtered_importance_scores = [score for feature, score in zip(best_features, importance_scores) if feature not in to_drop]

# Filtered features corresponding to the importance scores
filtered_features = [feature for feature in best_features if feature not in to_drop]

# Plotting the feature importance
plt.figure(figsize=(10, 6))
plt.barh(filtered_features, filtered_importance_scores, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()


# In[ ]:


y_train.shape


# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]  # Derive input dimension of capsule from input shape
        if self.share_weights:
            self.W = self.add_weight(
                shape=(1, self.num_capsule, input_dim_capsule, self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True,
                name='capsule_weights'
            )

    def squash(self, vectors):
        squared_norm = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
        scalar_factor = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + tf.keras.backend.epsilon())
        return scalar_factor * vectors

def call(self, u_vecs):
    # Expand dimensions to convert u_vecs to rank 3 tensor
    u_vecs = tf.expand_dims(u_vecs, axis=1)
    
    input_num_capsule = u_vecs.shape[1]
    if self.share_weights:
        u_hat_vecs = tf.expand_dims(u_vecs, -2)  # Expand the dimensions to add a new axis
        u_hat_vecs = tf.tile(u_hat_vecs, [1, self.num_capsule, 1, 1])  # Tile along the second axis
        u_hat_vecs = tf.expand_dims(u_hat_vecs, -1)  # Expand the dimensions to add a new axis
        u_hat_vecs = tf.tile(u_hat_vecs, [1, 1, 1, self.dim_capsule])  # Tile along the fourth axis
        u_hat_vecs = tf.reduce_sum(u_hat_vecs * self.W, axis=2)  # Perform element-wise multiplication and sum along the third axis
        u_hat_vecs = self.squash(u_hat_vecs)
    else:
        # Your existing code for handling separate weights for each capsule
        pass
    return u_hat_vecs


# Define Attention Layer
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        logits = tf.matmul(inputs, self.W)
        attention_weights = tf.nn.softmax(logits, axis=1)
        weighted_sum = tf.reduce_sum(inputs * attention_weights, axis=1)
        return weighted_sum



# In[ ]:


# Define input layers for CNN, GRU, and LSTM models
cnn_input = layers.Input(shape=(X_train_filtered.shape[1], 1))
gru_input = layers.Input(shape=(X_train_filtered.shape[1], 1))
lstm_input = layers.Input(shape=(X_train_filtered.shape[1], 1))

# Define your CNN model
cnn_model = models.Sequential([
    layers.Conv1D(filters=16, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),
    layers.Conv1D(filters=8, kernel_size=3, activation='relu')
])

# Define your GRU model
gru_model = models.Sequential([
    layers.GRU(units=16, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(8)  # Add a Dense layer after GRU if needed
])

# Define your LSTM model
lstm_model = models.Sequential([
    layers.LSTM(units=16, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(8)  # Add a Dense layer after LSTM if needed
])

# Apply CNN model to cnn_input
cnn_output = cnn_model(cnn_input)

# Apply GRU model to gru_input and reshape/repeat the output to match the temporal dimension of the CNN output
gru_output_matched = layers.RepeatVector(cnn_output.shape[1])(gru_model(gru_input))

# Apply LSTM model to lstm_input and reshape/repeat the output to match the temporal dimension of the CNN output
lstm_output_matched = layers.RepeatVector(cnn_output.shape[1])(lstm_model(lstm_input))

# Concatenate the outputs
merged_output = layers.Concatenate(axis=-1)([cnn_output, gru_output_matched, lstm_output_matched])

# Add Attention Layer
#attention_output = AttentionLayer()(merged_output)

# Add CapsuleLayer
capsule_layer_output = CapsuleLayer(num_capsule=4, dim_capsule=4, routings=3)(merged_output)

# Additional layers if needed
output = layers.Flatten()(capsule_layer_output)
output = layers.Dense(32, activation='relu')(output)
output = layers.Dropout(0.2)(output)
output = layers.Dense(y_train.shape[1], activation='softmax')(output)

# Create the final model
final_model = models.Model(inputs=[cnn_input, gru_input, lstm_input], outputs=output)

# Compile the model
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


# In[ ]:





# In[ ]:





# In[ ]:


#Define a learning rate schedular
def scheduler(epoch, lr):
    if epoch <=2:
        return lr #Keep the initial learning rate for the first 200 epochs
    else:
        return lr * tf.math.exp(-0.1) # Reduce the learning rate by a factor of 0.1 after the 200th epoch


# In[ ]:


lr=0.001
for i in range (10):
    lr=scheduler(i,lr)
    print(i,lr)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',  # monitor validation loss
                               patience=15,         # number of epochs to wait before stopping
                               restore_best_weights=True)


# In[ ]:


callback=LearningRateScheduler(scheduler)


# In[ ]:


# Train the model
#history = final_model.fit([X_train_filtered, X_train_reshaped, X_train_reshaped], y_train, validation_data=([X_val_filtered, X_val_reshaped, X_val_reshaped], y_val),callbacks=[early_stopping,callback], epochs=20, batch_size=64, verbose=2)


# In[ ]:


# Train the model
history = final_model.fit([X_train_filtered, X_train_filtered, X_train_filtered], y_train, validation_data=([X_val_filtered, X_val_filtered, X_val_filtered], y_val), callbacks=[early_stopping,callback], epochs=50, batch_size=64, verbose=2)


# In[ ]:


import numpy as np

# Parameters for complexity calculation
sequence_length = X_train_filtered.shape[1]  # Length of input sequence
cnn_filters1 = 16
cnn_kernel_size1 = 5
cnn_filters2 = 8
cnn_kernel_size2 = 3
gru_units = 16
lstm_units = 16
dense1_units = 32
num_classes = y_train.shape[1]
num_capsules = 4
dim_capsule = 4
routing_iterations = 3

# CNN Complexity
# Conv1D: O(sequence_length * filters * kernel_size)
cnn_complexity = (sequence_length * cnn_filters1 * cnn_kernel_size1) + \
                 (sequence_length // 2 * cnn_filters2 * cnn_kernel_size2)

# GRU Complexity
# GRU: O(sequence_length * units^2)
gru_complexity = sequence_length * gru_units**2

# LSTM Complexity
# LSTM: O(sequence_length * units^2)
lstm_complexity = sequence_length * lstm_units**2

# Capsule Layer Complexity
# Capsule Layer: O(num_capsules * dim_capsule * routing_iterations)
capsule_complexity = num_capsules * dim_capsule * routing_iterations

# Dense Layers Complexity
# Dense layer after flatten: O(input_size * output_size)
dense_complexity = (sequence_length * dim_capsule) + (dim_capsule * dense1_units) + (dense1_units * num_classes)

# Total complexity
total_complexity = cnn_complexity + gru_complexity + lstm_complexity + capsule_complexity + dense_complexity

# Print computational complexity
print(f"Computational Complexity of CNN: O({cnn_complexity})")
print(f"Computational Complexity of GRU: O({gru_complexity})")
print(f"Computational Complexity of LSTM: O({lstm_complexity})")
print(f"Computational Complexity of Capsule Layer: O({capsule_complexity})")
print(f"Computational Complexity of Dense Layers: O({dense_complexity})")
print(f"Total Computational Complexity: O({total_complexity})")


# In[ ]:


import numpy as np

# Parameters for complexity calculation
n = X_train_filtered.shape[1]  # Sequence length (input size)
cnn_filters1 = 16
cnn_kernel_size1 = 5
cnn_filters2 = 8
cnn_kernel_size2 = 3
gru_units = 16
lstm_units = 16
dense1_units = 32
num_classes = y_train.shape[1]
num_capsules = 4
dim_capsule = 4
routing_iterations = 3

# CNN Complexity
# Conv1D: O(n * filters * kernel_size)
cnn_complexity = (n * cnn_filters1 * cnn_kernel_size1) + (n // 2 * cnn_filters2 * cnn_kernel_size2)
cnn_complexity_big_o = f"O({cnn_filters1 * cnn_kernel_size1}n + {cnn_filters2 * cnn_kernel_size2}n/2)"

# GRU Complexity
# GRU: O(n * units^2)
gru_complexity = n * gru_units**2
gru_complexity_big_o = f"O({gru_units**2}n)"

# LSTM Complexity
# LSTM: O(n * units^2)
lstm_complexity = n * lstm_units**2
lstm_complexity_big_o = f"O({lstm_units**2}n)"

# Capsule Layer Complexity
# Capsule Layer: O(num_capsules * dim_capsule * routing_iterations)
capsule_complexity = num_capsules * dim_capsule * routing_iterations
capsule_complexity_big_o = f"O({num_capsules * dim_capsule * routing_iterations})"

# Dense Layers Complexity
# Dense: O(input_size * output_size)
dense_complexity = (n * dim_capsule) + (dim_capsule * dense1_units) + (dense1_units * num_classes)
dense_complexity_big_o = f"O({dim_capsule}n + {dim_capsule * dense1_units} + {dense1_units * num_classes})"

# Total complexity
total_complexity = cnn_complexity + gru_complexity + lstm_complexity + capsule_complexity + dense_complexity
total_complexity_big_o = f"O({cnn_filters1 * cnn_kernel_size1}n + {cnn_filters2 * cnn_kernel_size2}n/2 + {gru_units**2}n + {lstm_units**2}n + {num_capsules * dim_capsule * routing_iterations} + {dim_capsule}n + {dim_capsule * dense1_units} + {dense1_units * num_classes})"

# Print computational complexity
print(f"Computational Complexity of CNN: {cnn_complexity_big_o}")
print(f"Computational Complexity of GRU: {gru_complexity_big_o}")
print(f"Computational Complexity of LSTM: {lstm_complexity_big_o}")
print(f"Computational Complexity of Capsule Layer: {capsule_complexity_big_o}")
print(f"Computational Complexity of Dense Layers: {dense_complexity_big_o}")
print(f"Total Computational Complexity: {total_complexity_big_o}")


# In[ ]:


# Parameters for exact complexity calculation
n = X_train_filtered.shape[1]  # Sequence length (input size)
cnn_filters1 = 16
cnn_kernel_size1 = 5
cnn_filters2 = 8
cnn_kernel_size2 = 3
gru_units = 16
lstm_units = 16
dense1_units = 32
num_classes = y_train.shape[1]
num_capsules = 4
dim_capsule = 4
routing_iterations = 3

# CNN Complexity
# Conv1D complexity: sequence_length * filters * kernel_size
cnn_layer1_complexity = n * cnn_filters1 * cnn_kernel_size1
cnn_layer2_complexity = (n // 2) * cnn_filters2 * cnn_kernel_size2  # After max pooling
cnn_total_complexity = cnn_layer1_complexity + cnn_layer2_complexity

# GRU Complexity
# GRU: sequence_length * units^2
gru_complexity = n * gru_units**2

# LSTM Complexity
# LSTM: sequence_length * units^2
lstm_complexity = n * lstm_units**2

# Capsule Layer Complexity
# Capsule Layer: num_capsules * dim_capsule * routing_iterations
capsule_complexity = num_capsules * dim_capsule * routing_iterations

# Dense Layers Complexity
# Dense layers: input_size * output_size
dense_after_flatten = dim_capsule * n  # Flattened input from capsule layer
dense1_complexity = dense_after_flatten * dense1_units
dense2_complexity = dense1_units * num_classes
dense_total_complexity = dense1_complexity + dense2_complexity

# Total complexity
total_complexity = cnn_total_complexity + gru_complexity + lstm_complexity + capsule_complexity + dense_total_complexity

# Print exact number of operations for each part of the model
print(f"Exact Computational Complexity of Conv1D (layer 1): {cnn_layer1_complexity} operations")
print(f"Exact Computational Complexity of Conv1D (layer 2): {cnn_layer2_complexity} operations")
print(f"Exact Computational Complexity of CNN total: {cnn_total_complexity} operations")
print(f"Exact Computational Complexity of GRU: {gru_complexity} operations")
print(f"Exact Computational Complexity of LSTM: {lstm_complexity} operations")
print(f"Exact Computational Complexity of Capsule Layer: {capsule_complexity} operations")
print(f"Exact Computational Complexity of Dense layers: {dense_total_complexity} operations")
print(f"Total Exact Computational Complexity: {total_complexity} operations")


# In[ ]:


# Define symbolic variables
n = 'n'  # Sequence length
f1, f2 = 'f1', 'f2'  # Filters in CNN layers
k1, k2 = 'k1', 'k2'  # Kernel sizes in CNN layers
u_gru, u_lstm = 'u_GRU', 'u_LSTM'  # Units in GRU and LSTM
c, d, r = 'c', 'd', 'r'  # Capsule parameters: num_capsules, dim_capsule, routing_iterations
d1, m = 'd1', 'm'  # Dense layer units and output classes

# Construct complexity expressions
cnn_complexity = f"O({n} * {f1} * {k1} + {n}/2 * {f2} * {k2})"
gru_complexity = f"O({n} * {u_gru}^2)"
lstm_complexity = f"O({n} * {u_lstm}^2)"
capsule_complexity = f"O({c} * {d} * {r})"
dense_complexity = f"O({n} * {d} * {d1} + {d1} * {m})"

# Total complexity expression
total_complexity = f"O({n} * {f1} * {k1} + {n}/2 * {f2} * {k2} + {n} * {u_gru}^2 + {n} * {u_lstm}^2 + {c} * {d} * {r} + {n} * {d} * {d1} + {d1} * {m})"

# Print each part of the complexity
print(f"Computational Complexity of CNN: {cnn_complexity}")
print(f"Computational Complexity of GRU: {gru_complexity}")
print(f"Computational Complexity of LSTM: {lstm_complexity}")
print(f"Computational Complexity of Capsule Layer: {capsule_complexity}")
print(f"Computational Complexity of Dense Layers: {dense_complexity}")
print(f"Total Computational Complexity: {total_complexity}")


# In[ ]:





# In[ ]:





# In[ ]:


#history = model.fit(X_train_filtered, y_train, validation_data=(X_val_filtered, y_val), epochs=500, batch_size=64, callbacks=callback, verbose=2)


# In[ ]:


#history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_val, y_val), callbacks=[callback, early_stopping], verbose =1)
#history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=callback, verbose =1)


# In[ ]:


#class_weights = {0:0.5, 1:2.0,}


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

#y_test_pred = final_model.predict([X_test_filtered, X_test_gru, X_test_lstm])
y_test_pred = final_model.predict([X_test_filtered, X_test_filtered, X_test_filtered])
#y_test_pred =final_model.predict(X_test_filtered,X_test_filtered,X_test_filtered)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# 1. Check the shape of y_val and y_val_pred
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of y_test_pred: {y_test_pred.shape}")

# 2. Ensure y_val contains multiple classes
unique_classes = np.unique(y_test)
if len(unique_classes) < 2:
    raise ValueError("Only one unique class present in y_test. Check your data.")

# 3. Calculate and print accuracy, precision, recall, and ROC Curve for the entire dataset
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_test_pred_binary)
precision = precision_score(y_test, y_test_pred_binary, average='weighted')
recall = recall_score(y_test, y_test_pred_binary, average='weighted')

# 4. Calculate ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_test_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 5. Plot ROC Curve for each class
plt.figure(figsize=(8, 6))
for i in range(y_test.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# 2. Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test, y_test_pred_binary)


# In[ ]:


from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming multilabel_conf_matrix is the computed confusion matrix
for i in range(y_test.shape[1]):
    class_label = f'Class {i}'
    conf_matrix = multilabel_confusion_matrix(y_test[:, i], y_test_pred_binary[:, i])
    
    print(f"\nConfusion Matrix for {class_label}:\n{conf_matrix}")
    
    # Plot heatmap
    plt.figure(figsize=(6, 4))
    
    # For binary classification, consider only the first 2x2 matrix
    if conf_matrix.shape[0] == 2:
        sns.heatmap(conf_matrix[0], annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    else:
        # For multi-label, iterate through the individual confusion matrices
        for idx, cm in enumerate(conf_matrix):
            plt.subplot(1, len(conf_matrix), idx + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title(f'Sub-confusion matrix for {class_label} ({idx})')
    
    plt.title(f'Confusion Matrix for {class_label}')
    plt.show()


# In[ ]:


# Sum the confusion matrices along the first axis to aggregate them
agg_cm = np.sum(multilabel_conf_matrix, axis=0)

# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(agg_cm, annot=True, cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title('Aggregated Confusion Matrix')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report

# Calculate overall classification report
overall_report = classification_report(y_true, y_pred)

# Sum the confusion matrices along the first axis to aggregate them
agg_cm = np.sum(multilabel_conf_matrix, axis=0)

# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(agg_cm, annot=True, cmap='Blues', xticklabels=['True', 'False'], yticklabels=['True', 'False'])
plt.title('Aggregated Confusion Matrix')
plt.show()

# Print overall classification report
print("Overall Classification Report:")
print(overall_report)


# In[ ]:


# Print multi-label confusion matrix for each class with labels and heatmap
for i, class_label in enumerate(multilabel_conf_matrix.classes_):
    print(f"\nConfusion Matrix for Class '{class_label}':")
    print(multilabel_conf_matrix[i])

    # Plot heatmap for the current class
    plt.figure(figsize=(8, 6))
    #sns.heatmap(multilabel_conf_matrix[i], annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    sns.heatmap(multilabel_conf_matrix[i], annot=True,  cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix for Class \'{class_label}\'')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# In[ ]:


# 3. Print multi-label confusion matrix for each class with labels and heatmap
for i, multilabel_confusion_matrix in enumerate(multilabel_confusion_matrix):
    print(f"\nConfusion Matrix - Class {i}:\n", conf_matrix)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Multi-label Confusion Matrix - Class {i}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[ ]:


from sklearn.metrics import classification_report

# Assuming you have y_true (true labels) and y_pred (predicted labels)
# Calculate overall classification report
overall_report = classification_report(y_true, y_pred)

# Print overall classification report
print("Overall Classification Report:")
print(overall_report)

# Sum the confusion matrices along the first axis to aggregate them
agg_cm = np.sum(multilabel_conf_matrix, axis=0)

# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(agg_cm, annot=True, cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.title('Aggregated Confusion Matrix')
plt.show()


# In[ ]:


####*****************************
##$$$$$********************
#####7777*******************
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming class_names is a list containing the original class names
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Assuming multilabel_conf_matrix is the computed confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Print multi-label confusion matrix for each class with labels and heatmap
for i in range(len(class_names)):
    class_label = class_names[i]
    conf_matrix = multilabel_conf_matrix[i]

    print(f"\nConfusion Matrix - {class_label}:\n{conf_matrix}")

    # Plot heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Multi-label Confusion Matrix - {class_label}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot confusion matrix for each class
fig, axes = plt.subplots(6, 6, figsize=(15, 15))
fig.suptitle('Confusion Matrices for Each Class', fontsize=20)

for i, class_label in enumerate(class_names):
    row, col = divmod(i, 6)
    ax = axes[row, col]
    
    # Use scikit-learn's plot_confusion_matrix
    plot_confusion_matrix(None, y_val.argmax(axis=1) == i, y_val_pred.argmax(axis=1) == i, ax=ax, cmap='Blues')
    ax.set_title(class_label)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# In[ ]:


####*****************************
##$$$$$********************
#####7777*******************
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming class_names is a list containing the original class names
class_names = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood','DictionaryBruteForce','DNS_Spoofing','MITM-ArpSpoofing','DoS-SYN_Flood',
'DoS-UDP_Flood','DoS-TCP_Flood','DoS-HTTP_Flood','Recon-PingSweep','Recon-OSScan',
'VulnerabilityScan','Recon-PortScan','Recon-HostDiscovery','SqlInjection','CommandInjection','Backdoor_Malware',
'Uploading_Attack','XSS','BrowserHijacking','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain','BenignTraffic']

# Assuming multilabel_conf_matrix is the computed confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Print multi-label confusion matrix for each class with labels and heatmap
for i in range(len(class_names)):
    class_label = class_names[i]
    conf_matrix = multilabel_conf_matrix[i]

    print(f"\nConfusion Matrix - {class_label}:\n{conf_matrix}")

    # Plot heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Multi-label Confusion Matrix - {class_label}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[ ]:


#ALL Confusion Matrixxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming class_names is a list containing the original class names
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Assuming multilabel_conf_matrix is the computed confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

print("Multi-label Confusion Matrix:")
print(multilabel_conf_matrix)

# Print multi-label confusion matrix for each class with labels and heatmap
for i in range(len(class_names)):
    class_label = class_names[i]
    conf_matrix = multilabel_conf_matrix[i]

    print(f"\nConfusion Matrix - {class_label}:\n{conf_matrix}")

    # Plot heatmap
    plt.figure(figsize=(5, 4))
    #sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Multi-label Confusion Matrix - {class_label}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

from sklearn.metrics import classification_report

# Assuming y_val and y_val_pred are your multilabel true and predicted labels
unique_classes = np.unique(y_test.argmax(axis=1))
num_classes = len(unique_classes)
class_names = [f'Class {class_label}' for class_label in unique_classes]  # Modify this according to your class naming convention

classification_rep = classification_report(y_test.argmax(axis=1), y_test_pred.argmax(axis=1), labels=unique_classes, target_names=class_names)
print("Classification Report:\n", classification_rep)

# Assuming you have train_loss, train_accuracy, val_loss, and val_accuracy defined
#print("\nTraining Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test.argmax(axis=1), y_test_pred.argmax(axis=1), target_names=class_names))
#print("\nTraining Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Matthews Correlation Coefficient:", mcc)


# In[ ]:


from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming class_names is a list containing the original class names
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Assuming multilabel_conf_matrix is the computed confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

print("Multi-label Confusion Matrix:")
print(multilabel_conf_matrix)

# Print multi-label confusion matrix for each class with labels and heatmap
for i in range(len(class_names)):
    class_label = class_names[i]
    conf_matrix = multilabel_conf_matrix[i]

    print(f"\nConfusion Matrix - {class_label}:\n{conf_matrix}")

    # Plot heatmap
    plt.figure(figsize=(5, 4))
    #sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Multi-label Confusion Matrix - {class_label}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

from sklearn.metrics import classification_report

# Assuming y_val and y_val_pred are your multilabel true and predicted labels
unique_classes = np.unique(y_test.argmax(axis=1))
num_classes = len(unique_classes)
class_names = [f'Class {class_label}' for class_label in unique_classes]  # Modify this according to your class naming convention

classification_rep = classification_report(y_test.argmax(axis=1), y_test_pred.argmax(axis=1), labels=unique_classes, target_names=class_names)
print("Classification Report:\n", classification_rep)

# Assuming you have train_loss, train_accuracy, val_loss, and val_accuracy defined
#print("\nTraining Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test.argmax(axis=1), y_test_pred.argmax(axis=1), target_names=class_names))
#print("\nTraining Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Matthews Correlation Coefficient:", mcc)


# In[ ]:


# Print the results
print("Multilabel Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(multilabel_confusion_matrix.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))
#print(classification_report(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), target_names=mlb.classes_))
print("\nTraining Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Matthews Correlation Coefficient:", mcc)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=0), y_test_pred.argmax(axis=0))

# Print multi-label confusion matrix for each class with labels and heatmap
for i, class_label in enumerate(multilabel_conf_matrix.classes_):
    print(f"\nConfusion Matrix for Class '{class_label}':")
    print(multilabel_conf_matrix[i])

    # Plot heatmap for the current class
    plt.figure(figsize=(8, 6))
    #sns.heatmap(multilabel_conf_matrix[i], annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    sns.heatmap(multilabel_conf_matrix[i], annot=True,  cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix for Class \'{class_label}\'')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_test.argmax(axis=0),
    y_test_pred.argmax(axis=0),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))


# In[ ]:


# 4. Print overall classification report
class_labels = [f'Class {i}' for i in range(y_test.shape[1])]
classification_rep = classification_report(y_test, y_test_pred_binary, target_names=class_labels)
print("Classification Report:\n", classification_rep)


# In[ ]:


# 5. Sum the confusion matrices along the first axis to aggregate them
combined_confusion_matrix = np.sum(, axis=0)
# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(combined_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Aggregated Multi-label Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:


# 6. Plot ROC curve for each class with the Benign traffic class
benign_class_index = 0

plt.figure(figsize=(8, 6))
for i in range(y_val.shape[1]):
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    label = f'Class {i} (AUC = {roc_auc:.2f})'
    linestyle = '--' if i == benign_class_index else '-'
    plt.plot(fpr, tpr, label=label, linestyle=linestyle)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class with Benign Traffic Class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


benign_class_index = 0

plt.figure(figsize=(12, 8))
for i in range(y_val.shape[1]):
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Class {i} vs. Benign (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class with Benign Traffic Class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


benign_class_index = 0

plt.figure(figsize=(12, 8))
for i in range(y_val.shape[1]):
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Class {i} vs. Benign (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class with Benign Traffic Class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#########Gives two Benigns
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot ROC curve for each class along with Benign Traffic class
plt.figure(figsize=(10, 8))

for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

# Plot ROC curve for Benign Traffic class (assuming it's the last class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, linestyle='--', label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class with Benign Traffic')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


#Graphs ALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve(y_val[:, -1], y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve(y_val[:, -1], y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve(y_val[:, -1], y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve(y_val[:, -1], y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:





# In[ ]:


#################
#############This one i want
##########
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Define indices for "BenignTraffic" and attack classes
BenignTraffic_index = class_names.index('BenignTraffic')
attack_indices = [i for i, cls in enumerate(class_names) if cls != 'BenignTraffic']

# Aggregate predictions for all attack classes
y_test_attack = np.sum(y_test[:, attack_indices], axis=1)
y_test_pred_attack = np.sum(y_test_pred[:, attack_indices], axis=1)

# Compute ROC curve and AUC for combined attacks
fpr_attack, tpr_attack, _ = roc_curve(y_test_attack, y_test_pred_attack)
roc_auc_attack = auc(fpr_attack, tpr_attack)

# Compute ROC curve and AUC for BenignTraffic class
fpr_BenignTraffic, tpr_BenignTraffic, _ = roc_curve(y_test[:, BenignTraffic_index], y_test_pred[:, BenignTraffic_index])
roc_auc_BenignTraffic = auc(fpr_BenignTraffic, tpr_BenignTraffic)

# Plot ROC curves for combined attacks and BenignTraffic class on the same plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_attack, tpr_attack, color='darkorange', lw=2, label=f'Attacks (AUC = {roc_auc_attack:.4f})')
plt.plot(fpr_BenignTraffic, tpr_BenignTraffic, color='blue', lw=2, label=f'BenignTraffic (AUC = {roc_auc_BenignTraffic:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Attacks vs. BenignTraffic')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All the Attack Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


#################
#############This one i want
##########
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']          

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Define indices for "BenignTraffic" and attack classes
normal_index = class_names.index('Normal')
attack_indices = [i for i, cls in enumerate(class_names) if cls != 'Normal']

# Aggregate predictions for all attack classes
y_test_attack = np.sum(y_test[:, attack_indices], axis=1)
y_test_pred_attack = np.sum(y_test_pred[:, attack_indices], axis=1)

# Compute ROC curve and AUC for combined attacks
fpr_attack, tpr_attack, _ = roc_curve(y_test_attack, y_test_pred_attack)
roc_auc_attack = auc(fpr_attack, tpr_attack)

# Compute ROC curve and AUC for BenignTraffic class
fpr_normal, tpr_normal, _ = roc_curve(y_test[:, normal_index], y_test_pred[:, normal_index])
roc_auc_normal = auc(fpr_normal, tpr_normal)

# Plot ROC curves for combined attacks and BenignTraffic class on the same plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_attack, tpr_attack, color='darkorange', lw=2, label=f'Attacks (AUC = {roc_auc_attack:.4f})')
plt.plot(fpr_normal, tpr_normal, color='green', lw=2, label=f'BenignTraffic (AUC = {roc_auc_normal:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Attacks vs. BenignTraffic')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All the Attack Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:


###################FFFFFFFFFFFFFFFine one
###################
######@@@@@@@@@@@@@@@@22
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Initialize lists to store fpr, tpr, and roc_auc for each class
all_fpr = []
all_tpr = []
all_roc_auc = []

# Plot individual ROC curve and print classification report for each class
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_test[:, i], y_test_pred[:, i])
    roc_auc = auc(fpr, tpr)
    
    # Append fpr, tpr, and roc_auc to lists
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)
    
    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve for All Classes')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


###################FFFFFFFFFFFFFFFine one
###################
######@@@@@@@@@@@@@@@@22
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Initialize lists to store fpr, tpr, and roc_auc for each class
all_fpr = []
all_tpr = []
all_roc_auc = []

# Plot individual ROC curve and print classification report for each class
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_test[:, i], y_test_pred[:, i])
    roc_auc = auc(fpr, tpr)
    
    # Append fpr, tpr, and roc_auc to lists
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)
    
    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve for All Classes')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


#############
######### Seven ROCs in one ROC Curve
############
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Define new classes
new_class_names = ['DDoS', 'DoS', 'Recon', 'BruteForce', 'Web-Based', 'Spoofing', 'Mirai', 'Benign']

# Define mapping from old classes to new classes
class_mapping = {
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DictionaryBruteForce': 'BruteForce',
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'DoS-SYN_Flood': 'DoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-HostDiscovery': 'Recon',
    'SqlInjection': 'Web-Based',
    'CommandInjection': 'Web-Based',
    'Backdoor_Malware': 'Web-Based',
    'Uploading_Attack': 'Web-Based',
    'XSS': 'Web-Based',
    'BrowserHijacking': 'Web-Based',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'BenignTraffic': 'Benign'
}

# Convert old class names to new class names in y_val and y_val_pred
y_test_new = np.array([class_mapping[class_names[i]] for i in y_test.argmax(axis=1)])
y_test_pred_new = np.array([class_mapping[class_names[i]] for i in y_test_pred.argmax(axis=1)])

# Calculate multilabel confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test_new, y_test_pred_new, labels=new_class_names)

# Initialize lists to store fpr, tpr, and roc_auc for each class
all_fpr = []
all_tpr = []
all_roc_auc = []

# Plot individual ROC curve and print classification report for each class
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(new_class_names):
    # Aggregate predictions for the current class
    y_true = (y_test_new == class_label)
    y_pred = (y_test_pred_new == class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Append fpr, tpr, and roc_auc to lists
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)
    
    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve for All Classes')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All Classes:")
print(classification_report(
    y_test_new,
    y_test_pred_new,
    labels=new_class_names
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_test' (ground truth) and 'y_test_pred' (predicted values) defined somewhere

# Define new classes
new_class_names = ['DDoS', 'DoS', 'Recon', 'BruteForce', 'Web-Based', 'Spoofing', 'Mirai', 'Benign']

# Define mapping from old classes to new classes
class_mapping = {
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DictionaryBruteForce': 'BruteForce',
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'DoS-SYN_Flood': 'DoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-HostDiscovery': 'Recon',
    'SqlInjection': 'Web-Based',
    'CommandInjection': 'Web-Based',
    'Backdoor_Malware': 'Web-Based',
    'Uploading_Attack': 'Web-Based',
    'XSS': 'Web-Based',
    'BrowserHijacking': 'Web-Based',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'BenignTraffic': 'Benign'
}

# Initialize lists to store converted labels
y_test_new = []
y_test_pred_new = []

# Convert old class names to new class names in y_test
for i in y_test_argmax:
    class_name = class_names[i]
    if class_name in class_mapping:
        y_test_new.append(class_mapping[class_name])
    else:
        # Handle unknown class labels here (e.g., assign them a default label or ignore them)
        pass

# Convert old class names to new class names in y_test_pred
for i in y_test_pred_argmax:
    class_name = class_names[i]
    if class_name in class_mapping:
        y_test_pred_new.append(class_mapping[class_name])
    else:
        # Handle unknown class labels here (e.g., assign them a default label or ignore them)
        pass

# Calculate multilabel confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test_new, y_test_pred_new, labels=new_class_names)

# Initialize lists to store fpr, tpr, and roc_auc for each class
all_fpr = []
all_tpr = []
all_roc_auc = []

# Plot individual ROC curve and print classification report for each class
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(new_class_names):
    # Aggregate predictions for the current class
    y_true = (np.array(y_test_new) == class_label)
    y_pred = (np.array(y_test_pred_new) == class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Append fpr, tpr, and roc_auc to lists
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)
    
    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curve for All Classes')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All Classes:")
print(classification_report(
    y_test_new,
    y_test_pred_new,
    labels=new_class_names
))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


###############
################ This i want
################
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Define indices for "normal" and attack classes
normal_index = class_names.index('Normal')
attack_indices = [i for i, cls in enumerate(class_names) if cls != 'Normal']

# Aggregate predictions for all attack classes
y_test_attack = np.sum(y_test[:, attack_indices], axis=1)
y_test_pred_attack = np.sum(y_test_pred[:, attack_indices], axis=1)

# Compute ROC curve and AUC for combined attacks
fpr_attack, tpr_attack, _ = roc_curve(y_test_attack, y_test_pred_attack)
roc_auc_attack = auc(fpr_attack, tpr_attack)

# Compute ROC curve and AUC for normal class
fpr_normal, tpr_normal, _ = roc_curve(y_test[:, normal_index], y_test_pred[:, normal_index])
roc_auc_normal = auc(fpr_normal, tpr_normal)

# Plot ROC curves for combined attacks and normal class on the same plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_attack, tpr_attack, color='darkorange', lw=2, label=f'Attacks (AUC = {roc_auc_attack:.4f})')
plt.plot(fpr_normal, tpr_normal, color='green', lw=2, label=f'Normal (AUC = {roc_auc_normal:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Attacks vs. Normal')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All the Attack Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    target_names=class_names,
    output_dict=True
))



# In[ ]:





# In[ ]:


#############################
###########################
###########################
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with normal class")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
normal_index = 0  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_test[:, normal_index] == 1).astype(int), y_test_pred[:, normal_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[normal_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for normal Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))



# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test, y_test_pred)

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_test[:, i], y_test_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve(y_test[:, -1], y_test_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for normal')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Normal')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_test[:, i],
        y_test_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['normal','ddos','password','xss','injection','dos','scanning','mitm']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Initialize lists to store fpr, tpr, and roc_auc for each class
all_fpr = []
all_tpr = []
all_roc_auc = []

# Plot individual ROC curve and print classification report for each class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    
    # Append fpr, tpr, and roc_auc to lists
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

# Compute micro-average ROC curve and AUC
fpr_micro, tpr_micro, _ = roc_curve(y_val.ravel(), y_val_pred.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Plot micro-average ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_micro, tpr_micro, color='darkorange', lw=2, label=f'Weighted-average ROC curve (AUC = {roc_auc_micro:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Attack Classes')
plt.legend(loc='lower right')
plt.show()

# Print classification report for all classes
print("\nClassification Report for All the Attack Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val, y_val_pred)

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve(y_val[:, -1], y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Ensure that both y_val and y_val_pred have the same number of samples
assert y_val.shape[0] == y_val_pred.shape[0], "Number of samples in y_val and y_val_pred must be the same"

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val, y_val_pred)

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve(y_val[:, -1], y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        y_val[:, i],
        y_val_pred[:, i] >= 0.5,
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        (y_val[:, i] == 1).astype(int),
        (y_val_pred[:, i] >= 0.5).astype(int),
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        (y_val[:, i] == 1).astype(int),
        (y_val_pred[:, i] >= 0.5).astype(int),
        target_names=['Not ' + class_label, class_label],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        (y_val[:, i] == 1).astype(int),
        (y_val_pred[:, i] >= 0.5).astype(int),
        target_names=[f'Not {class_label}', f'{class_label}'],
        output_dict=True
    ))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood',
               'Recon-PingSweep', 'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery',
               'SqlInjection', 'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual ROC curve and print classification report for each class along with Benign Traffic class
for i in range(y_val.shape[1]):  # Iterate through all columns
    class_label = class_names[i]
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic class (assuming it's the last class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, -1] == 1).astype(int), y_val_pred[:, -1])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

    # Print classification report for the current class
    print(f"\nClassification Report for Class '{class_label}':")
    print(classification_report(
        (y_val[:, i] == 1).astype(int),
        (y_val_pred[:, i] >= 0.5).astype(int),
        target_names=[f'Not {class_label}', f'{class_label}'],
        output_dict=True
    ))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####################################These visulas look to me appealing if we consider the consistency of the Benign Traffic
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood','DictionaryBruteForce','DNS_Spoofing','MITM-ArpSpoofing','DoS-SYN_Flood'
'DoS-UDP_Flood','DoS-TCP_Flood','DoS-HTTP_Flood','Recon-OSScan',
'VulnerabilityScan','Recon-PortScan','Recon-HostDiscovery','SqlInjection','CommandInjection','Backdoor_Malware',
'Uploading_Attack','XSS','BrowserHijacking','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain','BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot ROC curve for each class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic (assuming it's the first class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood','DictionaryBruteForce','DNS_Spoofing','MITM-ArpSpoofing','DoS-SYN_Flood'
'DoS-UDP_Flood','DoS-TCP_Flood','Recon-PingSweep','Recon-OSScan',
'VulnerabilityScan','Recon-PortScan','Recon-HostDiscovery','SqlInjection','CommandInjection','Backdoor_Malware',
'Uploading_Attack','XSS','BrowserHijacking','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain','BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot ROC curve for each class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic (assuming it's the first class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names, replace this with the actual class names in your dataset
class_names = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood','DictionaryBruteForce','DNS_Spoofing','MITM-ArpSpoofing','DoS-SYN_Flood'
'DoS-UDP_Flood','DoS-TCP_Flood','DoS-HTTP_Flood''Recon-PingSweep','Recon-OSScan',
'VulnerabilityScan','Recon-PortScan','Recon-HostDiscovery','SqlInjection','CommandInjection','Backdoor_Malware',
'Uploading_Attack','XSS','BrowserHijacking','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain','BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot ROC curve for each class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic (assuming it's the first class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:


##########
######DDoS
######
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_test[:, 0] == 1).astype(int), y_test_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_test[:, class_index] == 1).astype(int), y_test_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', )

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for DDoS Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of classes to be aggregated into DDoS class
selected_classes = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
                    'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
                    'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for aggregated DDoS class
aggregated_ddos_label = np.sum(y_val[:, class_names.index(class_label)] for class_label in selected_classes)
aggregated_ddos_pred = np.sum(y_val_pred[:, class_names.index(class_label)] for class_label in selected_classes)
fpr_ddos, tpr_ddos, _ = roc_curve(aggregated_ddos_label, aggregated_ddos_pred)
roc_auc_ddos = auc(fpr_ddos, tpr_ddos)
plt.plot(fpr_ddos, tpr_ddos, color='red', linestyle='--', lw=2, label=f'DDoS (AUC = {roc_auc_ddos:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', )

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for DDoS Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Convert aggregated DDoS labels and predictions to binary for classification report
binary_ddos_label = (aggregated_ddos_label > 0).astype(int)
binary_ddos_pred = (aggregated_ddos_pred > 0).astype(int)

# Print overall classification report for aggregated DDoS class
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    binary_ddos_label,
    binary_ddos_pred,
    target_names=['DDoS', 'Benign Traffic']
))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


######Brute Force
#######
#######

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DictionaryBruteForce']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Brute Force Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


###########
##########Spoofing
#####
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DNS_Spoofing','MITM-ArpSpoofing']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Spoofing Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


###########DoS ROC
###########
############
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DoS-SYN_Flood','DoS-UDP_Flood','DoS-TCP_Flood','DoS-HTTP_Flood']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for DoS Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


###########
######Recon
#########
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Recon Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


######
####Web-Based
#####


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Web-Based Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


##########
########Mirai
#######

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_test[:, 0] == 1).astype(int), y_test_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_test[:, class_index] == 1).astype(int), y_test_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Web-Based Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Web-Based Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrix for selected classes
selected_conf_matrix = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        if index < len(multilabel_conf_matrix):
            selected_conf_matrix.append(multilabel_conf_matrix[index])
        else:
            print(f"Class '{class_label}' index is out of bounds. Skipping...")
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = sum(selected_conf_matrix)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrix for selected classes
selected_conf_matrix = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        if index < len(multilabel_conf_matrix):
            selected_conf_matrix.append(multilabel_conf_matrix[index])
        else:
            print(f"Class '{class_label}' index is out of bounds. Skipping...")
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = np.sum(selected_conf_matrix, axis=0)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Extract the confusion matrix for selected classes
selected_conf_matrix = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        if index < len(multilabel_conf_matrix):
            selected_conf_matrix.append(multilabel_conf_matrix[index])
        else:
            print(f"Class '{class_label}' index is out of bounds. Skipping...")
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = np.sum(selected_conf_matrix, axis=0)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrix for selected classes
selected_conf_matrix = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        if index < len(multilabel_conf_matrix):
            selected_conf_matrix.append(multilabel_conf_matrix[:, :, index])
        else:
            print(f"Class '{class_label}' index is out of bounds. Skipping...")
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = np.sum(selected_conf_matrix, axis=0)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrix for selected classes
selected_conf_matrix = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        if index < multilabel_conf_matrix.shape[0]:
            selected_conf_matrix.append(multilabel_conf_matrix[index])
        else:
            print(f"Class '{class_label}' index is out of bounds. Skipping...")
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = np.sum(selected_conf_matrix, axis=0)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate confusion matrix for each selected class
selected_conf_matrices = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[index, 32])
        selected_conf_matrices.append(conf_matrix)
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = np.sum(selected_conf_matrices, axis=0)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:





# In[ ]:


####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate confusion matrix for selected classes
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[class_names.index(class_label) for class_label in selected_classes])

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for DoS Category with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'BenignTraffic']

# Calculate confusion matrix for selected classes
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[class_names.index(class_label) for class_label in selected_classes])

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for DDoS Category with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DictionaryBruteForce', 'BenignTraffic']

# Calculate confusion matrix for selected classes
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[class_names.index(class_label) for class_label in selected_classes])

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Brute Force with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['DNS_Spoofing','MITM-ArpSpoofing', 'BenignTraffic']

# Calculate confusion matrix for selected classes
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[class_names.index(class_label) for class_label in selected_classes])

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Spoofing Category with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['Recon-PingSweep','Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'BenignTraffic']

# Calculate confusion matrix for selected classes
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[class_names.index(class_label) for class_label in selected_classes])

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Recon Category with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking', 'BenignTraffic']

# Calculate confusion matrix for selected classes
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[class_names.index(class_label) for class_label in selected_classes])

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Web-Based Category with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# List of selected classes (including BenignTraffic)
selected_classes = ['Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix for selected classes
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), labels=[class_names.index(class_label) for class_label in selected_classes])

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Mirai Category with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:





# In[ ]:


##########################Amazing
######################
##############
#########

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Define new classes
new_class_names = ['DDoS', 'DoS', 'Recon', 'Spoofing', 'BruteForce', 'Web-Based', 'Mirai', 'Benign']

# Define mapping from old classes to new classes
class_mapping = {
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DictionaryBruteForce': 'BruteForce',
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'DoS-SYN_Flood': 'DoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-HostDiscovery': 'Recon',
    'SqlInjection': 'Web-Based',
    'CommandInjection': 'Web-Based',
    'Backdoor_Malware': 'Web-Based',
    'Uploading_Attack': 'Web-Based',
    'XSS': 'Web-Based',
    'BrowserHijacking': 'Web-Based',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'BenignTraffic': 'Benign'
}

# Convert old class names to new class names in y_val and y_val_pred
y_test_new = np.array([class_mapping[class_names[i]] for i in y_test.argmax(axis=1)])
y_test_pred_new = np.array([class_mapping[class_names[i]] for i in y_test_pred.argmax(axis=1)])

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_new, y_test_pred_new, labels=new_class_names)

# Visualize the confusion matrix for new classes
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=new_class_names, yticklabels=new_class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Eight Classes")
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_test_new,
    y_test_pred_new,
    labels=new_class_names
))


# In[ ]:


##########################Amazing
######################
##############
#########

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Define new classes
new_class_names = ['DDoS', 'DoS', 'Recon', 'Spoofing', 'BruteForce', 'Web-Based', 'Mirai', 'Benign']

# Define mapping from old classes to new classes
class_mapping = {
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DictionaryBruteForce': 'BruteForce',
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'DoS-SYN_Flood': 'DoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-HostDiscovery': 'Recon',
    'SqlInjection': 'Web-Based',
    'CommandInjection': 'Web-Based',
    'Backdoor_Malware': 'Web-Based',
    'Uploading_Attack': 'Web-Based',
    'XSS': 'Web-Based',
    'BrowserHijacking': 'Web-Based',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'BenignTraffic': 'Benign'
}

# Convert old class names to new class names in y_val and y_val_pred
y_test_new = np.array([class_mapping[class_names[i]] for i in y_test.argmax(axis=1)])
y_test_pred_new = np.array([class_mapping[class_names[i]] for i in y_test_pred.argmax(axis=1)])

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_new, y_test_pred_new, labels=new_class_names)

# Visualize the confusion matrix for new classes
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=new_class_names, yticklabels=new_class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Eight Classes")
plt.show()
# Print overall classification report with additional metrics
classification_rep = classification_report(
    y_test_new,
    y_test_pred_new,
    labels=new_class_names,
    output_dict=True  # Set output_dict to True to get a dictionary output
)

# Print precision, recall, f1-score, false positive rate, false negative rate, and accuracy
for class_label in new_class_names:
    print(f"Class: {class_label}")
    print(f"Precision: {classification_rep[class_label]['precision']}")
    print(f"Recall: {classification_rep[class_label]['recall']}")
    print(f"F1-Score: {classification_rep[class_label]['f1-score']}")
    # Calculate false positive rate and false negative rate
    fp_rate = conf_matrix.sum(axis=0)[new_class_names.index(class_label)] - conf_matrix[new_class_names.index(class_label), new_class_names.index(class_label)]
    fp_rate /= conf_matrix.sum(axis=0).sum()
    fn_rate = conf_matrix.sum(axis=1)[new_class_names.index(class_label)] - conf_matrix[new_class_names.index(class_label), new_class_names.index(class_label)]
    fn_rate /= conf_matrix.sum(axis=1).sum()
    print(f"False Positive Rate: {fp_rate}")
    print(f"False Negative Rate: {fn_rate}")
    print("")

# Calculate overall accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print(f"Overall Accuracy: {accuracy}")


# In[ ]:


##########################Amazing
######################
##############
#########

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Define new classes
new_class_names = ['DDoS', 'DoS', 'Recon', 'Spoofing', 'BruteForce', 'Web-Based', 'Mirai', 'Benign']

# Define mapping from old classes to new classes
class_mapping = {
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DictionaryBruteForce': 'BruteForce',
    'DNS_Spoofing': 'Spoofing',
    'MITM-ArpSpoofing': 'Spoofing',
    'DoS-SYN_Flood': 'DoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Recon-PingSweep': 'Recon',
    'Recon-OSScan': 'Recon',
    'VulnerabilityScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-HostDiscovery': 'Recon',
    'SqlInjection': 'Web-Based',
    'CommandInjection': 'Web-Based',
    'Backdoor_Malware': 'Web-Based',
    'Uploading_Attack': 'Web-Based',
    'XSS': 'Web-Based',
    'BrowserHijacking': 'Web-Based',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'BenignTraffic': 'Benign'
}

# Convert old class names to new class names in y_val and y_val_pred
y_test_new = np.array([class_mapping[class_names[i]] for i in y_test.argmax(axis=1)])
y_test_pred_new = np.array([class_mapping[class_names[i]] for i in y_test_pred.argmax(axis=1)])

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_new, y_test_pred_new, labels=new_class_names)

# Visualize the confusion matrix for new classes
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=new_class_names, yticklabels=new_class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Eight Classes")
plt.show()
# Print overall classification report with additional metrics
classification_rep = classification_report(
    y_test_new,
    y_test_pred_new,
    labels=new_class_names,
    output_dict=True  # Set output_dict to True to get a dictionary output
)

# Print precision, recall, f1-score, false positive rate, false negative rate, true positive rate, true negative rate, and accuracy
for class_label in new_class_names:
    print(f"Class: {class_label}")
    print(f"Precision: {classification_rep[class_label]['precision']}")
    print(f"Recall: {classification_rep[class_label]['recall']}")
    print(f"F1-Score: {classification_rep[class_label]['f1-score']}")

    # Calculate true positive rate (TPR) and true negative rate (TNR)
    tn = conf_matrix.sum() - np.sum(conf_matrix[new_class_names.index(class_label)])
    tp = conf_matrix[new_class_names.index(class_label), new_class_names.index(class_label)]
    fn = conf_matrix.sum(axis=1)[new_class_names.index(class_label)] - tp
    fp = conf_matrix.sum(axis=0)[new_class_names.index(class_label)] - tp
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    
    print(f"True Positive Rate (TPR): {tpr}")
    print(f"True Negative Rate (TNR): {tnr}")
    
    # Calculate false positive rate (FPR) and false negative rate (FNR)
    fp_rate = fp / (fp + tn)
    fn_rate = fn / (fn + tp)
    print(f"False Positive Rate (FPR): {fp_rate}")
    print(f"False Negative Rate (FNR): {fn_rate}")
    
    print("")

# Calculate overall accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print(f"Overall Accuracy: {accuracy}")


# In[ ]:


# Initialize variables to store weighted metrics
weighted_accuracy = 0
weighted_precision = 0
weighted_recall = 0
weighted_f1_score = 0
weighted_tpr = 0
weighted_tnr = 0
weighted_fpr = 0
weighted_fnr = 0
total_samples = len(y_test_new)

# Calculate weighted metrics
for class_label in new_class_names:
    # Calculate metrics for the current class
    tn = conf_matrix.sum() - np.sum(conf_matrix[new_class_names.index(class_label)])
    tp = conf_matrix[new_class_names.index(class_label), new_class_names.index(class_label)]
    fn = conf_matrix.sum(axis=1)[new_class_names.index(class_label)] - tp
    fp = conf_matrix.sum(axis=0)[new_class_names.index(class_label)] - tp
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Update weighted metrics
    weighted_accuracy += (tp + tn) / total_samples
    weighted_precision += classification_rep[class_label]['precision'] * (tp + fn) / total_samples
    weighted_recall += classification_rep[class_label]['recall'] * (tp + fn) / total_samples
    weighted_f1_score += classification_rep[class_label]['f1-score'] * (tp + fn) / total_samples
    weighted_tpr += tpr * (tp + fn) / total_samples
    weighted_tnr += tnr * (tp + fn) / total_samples
    weighted_fpr += fpr * (tp + fn) / total_samples
    weighted_fnr += fnr * (tp + fn) / total_samples

# Print weighted metrics
print("Weighted Metrics:")
print(f"Weighted Accuracy: {weighted_accuracy}")
print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1-Score: {weighted_f1_score}")
print(f"Weighted TPR: {weighted_tpr}")
print(f"Weighted TNR: {weighted_tnr}")
print(f"Weighted FPR: {weighted_fpr}")
print(f"Weighted FNR: {weighted_fnr}")


# In[ ]:


# Initialize dictionaries to store weighted metrics for each category
weighted_metrics_per_category = {class_label: {} for class_label in new_class_names}

# Calculate weighted metrics for each category
for class_label in new_class_names:
    # Calculate metrics for the current class
    tn = conf_matrix.sum() - np.sum(conf_matrix[new_class_names.index(class_label)])
    tp = conf_matrix[new_class_names.index(class_label), new_class_names.index(class_label)]
    fn = conf_matrix.sum(axis=1)[new_class_names.index(class_label)] - tp
    fp = conf_matrix.sum(axis=0)[new_class_names.index(class_label)] - tp
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Calculate weighted metrics
    weighted_accuracy = (tp + tn) / len(y_test_new[y_test_new == class_label])
    weighted_precision = classification_rep[class_label]['precision']
    weighted_recall = classification_rep[class_label]['recall']
    weighted_f1_score = classification_rep[class_label]['f1-score']
    weighted_tpr = tpr
    weighted_tnr = tnr
    weighted_fpr = fpr
    weighted_fnr = fnr

    # Store weighted metrics for the current class
    weighted_metrics_per_category[class_label]['Weighted Accuracy'] = weighted_accuracy
    weighted_metrics_per_category[class_label]['Weighted Precision'] = weighted_precision
    weighted_metrics_per_category[class_label]['Weighted Recall'] = weighted_recall
    weighted_metrics_per_category[class_label]['Weighted F1-Score'] = weighted_f1_score
    weighted_metrics_per_category[class_label]['Weighted TPR'] = weighted_tpr
    weighted_metrics_per_category[class_label]['Weighted TNR'] = weighted_tnr
    weighted_metrics_per_category[class_label]['Weighted FPR'] = weighted_fpr
    weighted_metrics_per_category[class_label]['Weighted FNR'] = weighted_fnr

# Print weighted metrics for each category
for class_label in new_class_names:
    print(f"Weighted Metrics for {class_label}:")
    for metric_name, metric_value in weighted_metrics_per_category[class_label].items():
        print(f"{metric_name}: {metric_value}")
    print("")


# In[ ]:


# Initialize dictionaries to store weighted metrics for each category
weighted_metrics_per_category = {class_label: {} for class_label in new_class_names}

# Calculate weighted metrics for each category
for class_label in new_class_names:
    # Calculate metrics for the current class
    tn = conf_matrix.sum() - np.sum(conf_matrix[new_class_names.index(class_label)])
    tp = conf_matrix[new_class_names.index(class_label), new_class_names.index(class_label)]
    fn = conf_matrix.sum(axis=1)[new_class_names.index(class_label)] - tp
    fp = conf_matrix.sum(axis=0)[new_class_names.index(class_label)] - tp
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    # Calculate weighted metrics
    weighted_accuracy = (tp + tn) / len(y_test_new[y_test_new == class_label])
    weighted_precision = classification_rep[class_label]['precision']
    weighted_recall = classification_rep[class_label]['recall']
    weighted_f1_score = classification_rep[class_label]['f1-score']
    weighted_tpr = tpr
    weighted_tnr = tnr
    weighted_fpr = fpr
    weighted_fnr = fnr

    # Store weighted metrics for the current class
    weighted_metrics_per_category[class_label]['Weighted Accuracy'] = weighted_accuracy
    weighted_metrics_per_category[class_label]['Weighted Precision'] = weighted_precision
    weighted_metrics_per_category[class_label]['Weighted Recall'] = weighted_recall
    weighted_metrics_per_category[class_label]['Weighted F1-Score'] = weighted_f1_score
    weighted_metrics_per_category[class_label]['Weighted TPR'] = weighted_tpr
    weighted_metrics_per_category[class_label]['Weighted TNR'] = weighted_tnr
    weighted_metrics_per_category[class_label]['Weighted FPR'] = weighted_fpr
    weighted_metrics_per_category[class_label]['Weighted FNR'] = weighted_fnr

# Print weighted metrics for each category
for class_label in new_class_names:
    print(f"Weighted Metrics for {class_label}:")
    for metric_name, metric_value in weighted_metrics_per_category[class_label].items():
        print(f"{metric_name}: {metric_value}")
    print("")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of selected classes along with Benign Traffic
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrix for selected classes
selected_conf_matrix = multilabel_conf_matrix[[class_names.index(class_label) for class_label in selected_classes]]

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(selected_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of selected classes (excluding BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrix for selected classes
selected_conf_matrix = multilabel_conf_matrix[[class_names.index(class_label) for class_label in selected_classes]]

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(selected_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes (Excluding Benign Traffic)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of selected classes (excluding BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot individual confusion matrices for selected classes
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Extract the confusion matrix for the current class
    class_conf_matrix = multilabel_conf_matrix[class_index]
    
    # Plot the confusion matrix heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(class_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix for {class_label}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrices for selected classes
selected_conf_matrices = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        selected_conf_matrices.append(multilabel_conf_matrix[index])
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset.")

# Combine confusion matrices for selected classes
combined_conf_matrix = sum(selected_conf_matrices)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrices for selected classes
selected_conf_matrices = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        selected_conf_matrices.append(multilabel_conf_matrix[index])
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = sum(selected_conf_matrices)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrices for selected classes
selected_conf_matrices = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        selected_conf_matrices.append(multilabel_conf_matrix[index])
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = sum(selected_conf_matrices)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of selected classes (including BenignTraffic)
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'BenignTraffic']

# Ensure 'BenignTraffic' is present in class_names
if 'BenignTraffic' not in class_names:
    class_names.append('BenignTraffic')

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Extract the confusion matrices for selected classes
selected_conf_matrices = []
for class_label in selected_classes:
    try:
        index = class_names.index(class_label)
        selected_conf_matrices.append(multilabel_conf_matrix[index])
    except ValueError:
        print(f"Class '{class_label}' not found in the dataset. Skipping...")

# Combine confusion matrices for selected classes
combined_conf_matrix = sum(selected_conf_matrices)

# Plot the confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(combined_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=selected_classes, yticklabels=selected_classes)
plt.title('Confusion Matrix for Selected Classes with Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['SqlInjection','CommandInjection','Backdoor_Malware','Uploading_Attack','XSS','BrowserHijacking']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Web-Based Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['Mirai-greip_flood',
               'Mirai-greeth_flood',  'Mirai-udpplain', 'DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking','BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['Mirai-greip_flood','Mirai-greeth_flood','Mirai-udpplain']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(12, 8))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'Benign Traffic (AUC = {roc_auc_benign:.2f})')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'{class_label} (AUC = {roc_auc:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Mirai Category with Benign Traffic Class')

# Show the plot
plt.legend(loc='lower right')
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#THIS CODE IS FOR THE HEATMAP ChecK IT WHETHER IT WORKS OR NOT
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of class names from your dataset
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set the figure size explicitly using plt.figure()
plt.figure(figsize=(15, 10))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
# ... (same as before)

# Plot ROC curve for each selected class
# ... (same as before)

# Plot the diagonal line for reference
# ... (same as before)

# Set labels and title
# ... (same as before)

# Show the plot
# ... (same as before)

# Plot heatmap for the selected classes and Benign Traffic
plt.figure(figsize=(15, 10))
sns.heatmap(multilabel_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title('Multi-label Confusion Matrix for Selected Classes and Benign Traffic')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print overall classification report for selected classes
# ... (same as before)


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DoS-SYN_Flood', 'Recon-PingSweep', 'VulnerabilityScan', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot a single ROC curve for selected classes
plt.figure(figsize=(8, 6))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for  with Benign Traffic')
plt.legend(loc='lower right')

# Show the plot
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DoS-SYN_Flood', 'Recon-PingSweep', 'VulnerabilityScan', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot ROC curve for each selected class
fig, axs = plt.subplots(len(selected_classes), 1, figsize=(8, 6 * len(selected_classes)))

for i, class_label in enumerate(selected_classes):
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    axs[i].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

    # Plot ROC curve for Benign Traffic (assuming it's the first class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    axs[i].plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

    # Plot the diagonal line for reference
    axs[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Set labels and title for each subplot
    axs[i].set_xlabel('False Positive Rate')
    axs[i].set_ylabel('True Positive Rate')
    axs[i].set_title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    axs[i].legend(loc='lower right')

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# List of seven classes of your choice along with Benign Traffic
selected_classes = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DoS-SYN_Flood', 'Recon-PingSweep', 'VulnerabilityScan', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot a single ROC curve for selected classes
plt.figure(figsize=(8, 6))

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc_benign = auc(fpr_benign, tpr_benign)
plt.plot(fpr_benign, tpr_benign, color='green', linestyle='--', lw=2, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic')

# Plot ROC curve for each selected class
for class_label in selected_classes:
    class_index = class_names.index(class_label)
    
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, class_index] == 1).astype(int), y_val_pred[:, class_index])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Selected Classes with Benign Traffic')
plt.legend(loc='lower right')

# Show the plot
plt.show()

# Print overall classification report for selected classes
print("\nOverall Classification Report for Selected Classes:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=[class_names.index(class_label) for class_label in selected_classes],
    target_names=selected_classes,
    output_dict=True
))


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot the confusion matrix for Benign Traffic (assuming it's the first class)
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
benign_conf_matrix = multilabel_conf_matrix[benign_index]

# Visualize the confusion matrix using seaborn heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(benign_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Benign Traffic Class")
plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Visualize the entire confusion matrix using seaborn heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(multilabel_conf_matrix.sum(axis=0), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes")
plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='white')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Visualize the entire confusion matrix with all classes, including Benign Traffic
plt.figure(figsize=(12, 10))
sns.heatmap(multilabel_conf_matrix.sum(axis=0), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")
plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Visualize the entire confusion matrix with all classes, including Benign Traffic
plt.figure(figsize=(12, 10))
sns.heatmap(multilabel_conf_matrix.sum(axis=0), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")
plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(12, 10))
sns.heatmap(multilabel_conf_matrix.sum(axis=0), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=['Count'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")
plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")
plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 1  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_test[:, benign_index] == 1).astype(int), y_test_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$&&&&&&&&&&&&&&&&&&&&&&&&&&&
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")
plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_test[:, benign_index] == 1).astype(int), y_test_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:


###############Granularity check
#################
################

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_test' (ground truth) and 'y_test_pred' (predicted values)
# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Calculate metrics for each class
metrics_per_class = classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
)

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")
plt.show()

# Initialize dictionaries to store metrics
weighted_metrics = {'precision': [], 'recall': [], 'f1-score': [], 'tpr': [], 'tnr': [], 'fpr': [], 'fnr': []}

# Loop through each class to calculate metrics
for idx, class_name in enumerate(class_names):
    precision = metrics_per_class[class_name]['precision']
    recall = metrics_per_class[class_name]['recall']
    f1_score = metrics_per_class[class_name]['f1-score']
    tp = conf_matrix[idx, idx]
    fp = np.sum(conf_matrix[:, idx]) - tp
    fn = np.sum(conf_matrix[idx, :]) - tp
    tn = np.sum(conf_matrix) - tp - fp - fn
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    weighted_metrics['precision'].append(precision)
    weighted_metrics['recall'].append(recall)
    weighted_metrics['f1-score'].append(f1_score)
    weighted_metrics['tpr'].append(tpr)
    weighted_metrics['tnr'].append(tnr)
    weighted_metrics['fpr'].append(fpr)
    weighted_metrics['fnr'].append(fnr)

# Calculate weighted metrics
weighted_precision = np.average(weighted_metrics['precision'])
weighted_recall = np.average(weighted_metrics['recall'])
weighted_f1_score = np.average(weighted_metrics['f1-score'])
weighted_tpr = np.average(weighted_metrics['tpr'])
weighted_tnr = np.average(weighted_metrics['tnr'])
weighted_fpr = np.average(weighted_metrics['fpr'])
weighted_fnr = np.average(weighted_metrics['fnr'])

# Print metrics for each class
print("Metrics for Each Class:")
for idx, class_name in enumerate(class_names):
    print(f"Class: {class_name}")
    print(f"  Precision: {weighted_metrics['precision'][idx]:.4f}")
    print(f"  Recall: {weighted_metrics['recall'][idx]:.4f}")
    print(f"  F1-score: {weighted_metrics['f1-score'][idx]:.4f}")
    print(f"  TPR: {weighted_metrics['tpr'][idx]:.4f}")
    print(f"  TNR: {weighted_metrics['tnr'][idx]:.4f}")
    print(f"  FPR: {weighted_metrics['fpr'][idx]:.4f}")
    print(f"  FNR: {weighted_metrics['fnr'][idx]:.4f}")

# Print weighted metrics
print("\nWeighted Metrics:")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")
print(f"Weighted F1-score: {weighted_f1_score:.4f}")
print(f"Weighted TPR: {weighted_tpr:.4f}")
print(f"Weighted TNR: {weighted_tnr:.4f}")
print(f"Weighted FPR: {weighted_fpr:.4f}")
print(f"Weighted FNR: {weighted_fnr:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:


#############################
###########################
###########################
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['Normal','Generic','Exploits','Fuzzers','DoS','Reconnaissance','Analysis','Backdoor','Shellcode',
'Worms']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 1  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_test[:, benign_index] == 1).astype(int), y_test_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Display confusion matrix values for all classes, including Benign Traffic
print("\nConfusion Matrix:")
print(conf_matrix)

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = 32  # Change this index based on the actual index of BenignTraffic in your class_names list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = class_names.index('BenignTraffic')
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Add a row and column for Benign Traffic in the confusion matrix
conf_matrix_with_benign = np.zeros((conf_matrix.shape[0]+1, conf_matrix.shape[1]+1), dtype=conf_matrix.dtype)
conf_matrix_with_benign[:-1, :-1] = conf_matrix

# Set labels for the added row and column
class_names_with_benign = class_names + ['BenignTraffic']

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix_with_benign, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = class_names_with_benign.index('BenignTraffic')
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names_with_benign[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Add a row and column for Benign Traffic in the confusion matrix
benign_index = class_names.index('BenignTraffic')
conf_matrix[:, benign_index] = np.sum(conf_matrix[:, benign_index:benign_index+1], axis=1)
conf_matrix[benign_index, :] = np.sum(conf_matrix[benign_index:benign_index+1, :], axis=0)

# Set labels for the added row and column
class_names_with_benign = class_names + ['BenignTraffic']

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names_with_benign[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Add a row and column for Benign Traffic in the confusion matrix
conf_matrix = np.insert(conf_matrix, benign_index, 0, axis=0)
conf_matrix = np.insert(conf_matrix, benign_index, 0, axis=1)

# Sum true positive counts for the Benign Traffic class
conf_matrix[benign_index, benign_index] = np.sum(conf_matrix[:, benign_index])
conf_matrix[:, benign_index] = np.sum(conf_matrix[:, [benign_index, -1]], axis=1)

# Set labels for the added row and column
class_names_with_benign = class_names + ['BenignTraffic']

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names_with_benign[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Add a row and column for Benign Traffic in the confusion matrix
conf_matrix_with_benign = np.zeros((conf_matrix.shape[0] + 1, conf_matrix.shape[1] + 1), dtype=conf_matrix.dtype)
conf_matrix_with_benign[:-1, :-1] = conf_matrix

# Sum true positive counts for the Benign Traffic class
true_positive_count = np.sum(y_val[:, -1] == 1)
conf_matrix_with_benign[-1, -1] = true_positive_count

# Set labels for the added row and column
class_names_with_benign = class_names + ['BenignTraffic']

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix_with_benign, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = len(class_names_with_benign) - 1  # Index of 'BenignTraffic' in the extended list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names_with_benign[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Add a row and column for Benign Traffic in the confusion matrix
conf_matrix_with_benign = np.zeros((conf_matrix.shape[0] + 1, conf_matrix.shape[1] + 1), dtype=conf_matrix.dtype)
conf_matrix_with_benign[:-1, :-1] = conf_matrix

# Sum true positive counts for the Benign Traffic class
true_positive_count = np.sum(y_val[:, -1] == 1)
conf_matrix_with_benign[-1, -1] = true_positive_count

# Set labels for the added row and column
class_names_with_benign = class_names + ['BenignTraffic']

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix_with_benign, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = len(class_names_with_benign) - 1  # Index of 'BenignTraffic' in the extended list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names_with_benign[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Set labels for the confusion matrix
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Visualize the confusion matrix for all classes
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = len(class_names) - 1  # Index of 'BenignTraffic' in the list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Add a row and column for Benign Traffic in the confusion matrix
conf_matrix_with_benign = np.zeros((conf_matrix.shape[0] + 1, conf_matrix.shape[1] + 1), dtype=conf_matrix.dtype)
conf_matrix_with_benign[:-1, :-1] = conf_matrix

# Calculate true positive count for the Benign Traffic class
benign_true_positive_count = np.sum((y_val[:, -1] == 1) & (y_val_pred[:, -1] == 1))
conf_matrix_with_benign[-1, -1] = benign_true_positive_count

# Set labels for the added row and column
class_names_with_benign = class_names + ['BenignTraffic']

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix_with_benign, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
benign_index = len(class_names_with_benign) - 1  # Index of 'BenignTraffic' in the extended list
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names_with_benign[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Sum true positive counts for the Benign Traffic class
true_positive_count = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))
conf_matrix[benign_index, benign_index] = true_positive_count

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Sum true positive counts for the Benign Traffic class
true_positive_count = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))
# Update the correct entry in the confusion matrix
conf_matrix[benign_index, benign_index] = true_positive_count

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Print the found index
print("Index of 'BenignTraffic':", benign_index)

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Print the length of class names and confusion matrix size
print("Length of class names:", len(class_names))
print("Shape of confusion matrix:", conf_matrix.shape)

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Print the found index
print("Index of 'BenignTraffic':", benign_index)

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Print the new shape of the confusion matrix
print("New shape of confusion matrix:", conf_matrix.shape)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Check if 'BenignTraffic' is present in the predicted labels before attempting to remove it
if benign_index < conf_matrix.shape[0]:
    # Remove extra row and column for Benign Traffic without values
    conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
    conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

    # Set labels for the confusion matrix
    class_names_without_benign = class_names.copy()
    class_names_without_benign.remove('BenignTraffic')

    # Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

    # Set labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

    plt.show()

    # Initialize a figure for ROC curves
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for Benign Traffic
    fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Benign Traffic Class')
    plt.legend(loc='lower right')
    plt.show()

    # Print overall classification report
    print("\nOverall Classification Report:")
    print(classification_report(
        y_val.argmax(axis=1),
        y_val_pred.argmax(axis=1),
        labels=np.arange(len(class_names_without_benign)),
        target_names=class_names_without_benign,
        output_dict=True
    ))
else:
    print("BenignTraffic not present in predicted labels.")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Check if 'BenignTraffic' is present in the predicted labels before attempting to remove it
if benign_index < conf_matrix.shape[0]:
    # Remove row and column for Benign Traffic
    conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
    conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

    # Set labels for the confusion matrix
    class_names_without_benign = class_names.copy()
    class_names_without_benign.remove('BenignTraffic')

    # Visualize the confusion matrix for all classes, excluding the Benign Traffic
    plt.figure(figsize=(14, 12))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

    # Set labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for All Classes without Benign Traffic")

    plt.show()

    # Initialize a figure for ROC curves
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for Benign Traffic
    fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

    # Plot the diagonal line for reference
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Benign Traffic Class')
    plt.legend(loc='lower right')
    plt.show()

    # Print overall classification report
    print("\nOverall Classification Report:")
    print(classification_report(
        y_val.argmax(axis=1),
        y_val_pred.argmax(axis=1),
        labels=np.arange(len(class_names_without_benign)),
        target_names=class_names_without_benign,
        output_dict=True
    ))
else:
    print("BenignTraffic not present in predicted labels.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Remove extra row and column for Benign Traffic without values
conf_matrix = np.delete(conf_matrix, benign_index, axis=0)
conf_matrix = np.delete(conf_matrix, benign_index, axis=1)

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding the extra Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Extra Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Set labels for the confusion matrix
class_names_without_benign = class_names.copy()
class_names_without_benign.remove('BenignTraffic')

# Visualize the confusion matrix for all classes, excluding Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix[:-1, :-1], annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_without_benign, yticklabels=class_names_without_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes without Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_without_benign)),
    target_names=class_names_without_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Add a row and column for Benign Traffic in the confusion matrix
conf_matrix_with_benign = np.zeros((conf_matrix.shape[0] + 1, conf_matrix.shape[1] + 1), dtype=conf_matrix.dtype)
conf_matrix_with_benign[:-1, :-1] = conf_matrix

# Sum true positive counts for the Benign Traffic class
conf_matrix_with_benign[benign_index, benign_index] = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))

# Set labels for the added row and column
class_names_with_benign = class_names + ['BenignTraffic']

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix_with_benign, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Sum true positive counts for the Benign Traffic class
true_positive_count = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))
conf_matrix[benign_index, benign_index] = true_positive_count

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


######*****
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Sum true positive counts for the Benign Traffic class
true_positive_count = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))

# If the confusion matrix size is smaller than the index, extend it
if benign_index >= conf_matrix.shape[0]:
    conf_matrix = np.pad(conf_matrix, ((0, benign_index + 1 - conf_matrix.shape[0]), (0, benign_index + 1 - conf_matrix.shape[1])))

conf_matrix[benign_index, benign_index] = true_positive_count

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Calculate true positive counts for the Benign Traffic class
true_positive_count = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))

# If the confusion matrix size is smaller than the index, extend it
if benign_index >= conf_matrix.shape[0]:
    conf_matrix = np.pad(conf_matrix, ((0, benign_index + 1 - conf_matrix.shape[0]), (0, benign_index + 1 - conf_matrix.shape[1])))

conf_matrix[benign_index, benign_index] = true_positive_count

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, multilabel_confusion_matrix
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Calculate true positive counts for the Benign Traffic class
true_positive_count = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))[benign_index, 1, 1]

# If the confusion matrix size is smaller than the index, extend it
if benign_index >= conf_matrix.shape[0]:
    conf_matrix = np.pad(conf_matrix, ((0, benign_index + 1 - conf_matrix.shape[0]), (0, benign_index + 1 - conf_matrix.shape[1])))

conf_matrix[benign_index, benign_index] = true_positive_count

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, multilabel_confusion_matrix
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Calculate true positive counts for the Benign Traffic class
conf_matrix_multilabel = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))
true_positive_count = conf_matrix_multilabel[benign_index, 1, 1]

# If the confusion matrix size is smaller than the index, extend it
if benign_index >= conf_matrix.shape[0]:
    conf_matrix = np.pad(conf_matrix, ((0, benign_index + 1 - conf_matrix.shape[0]), (0, benign_index + 1 - conf_matrix.shape[1])))

conf_matrix[benign_index, benign_index] = true_positive_count

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


#####@@@@@@@@@@@@@@@@@@@@@@@@@@@
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, multilabel_confusion_matrix
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Calculate true positive counts for the Benign Traffic class
conf_matrix_multilabel = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))
true_positive_count = 0  # Initialize to 0
# If the index is within bounds, update the true positive count
if benign_index < conf_matrix_multilabel.shape[0]:
    true_positive_count = conf_matrix_multilabel[benign_index, 1, 1]

# If the confusion matrix size is smaller than the index, extend it
conf_matrix_size = min(conf_matrix.shape[0], conf_matrix_multilabel.shape[0])
if benign_index >= conf_matrix_size:
    conf_matrix = np.pad(conf_matrix, ((0, benign_index + 1 - conf_matrix_size), (0, benign_index + 1 - conf_matrix_size)))

# Set the correct entry for the true positive count
conf_matrix[benign_index, benign_index] = true_positive_count

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, multilabel_confusion_matrix
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Calculate true positives for the Benign Traffic class
true_positives = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))

# Update the correct entry in the confusion matrix
conf_matrix[benign_index, benign_index] = true_positives

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))



# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation', 'DDoS-UDP_Flood', 'DDoS-SlowLoris', 'DDoS-ICMP_Flood', 'DDoS-RSTFINFlood',
               'DDoS-PSHACK_Flood', 'DDoS-HTTP_Flood', 'DDoS-UDP_Fragmentation', 'DDoS-ICMP_Fragmentation',
               'DDoS-TCP_Flood', 'DDoS-SYN_Flood', 'DDoS-SynonymousIP_Flood', 'DictionaryBruteForce', 'DNS_Spoofing',
               'MITM-ArpSpoofing', 'DoS-SYN_Flood', 'DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-HTTP_Flood', 'Recon-PingSweep',
               'Recon-OSScan', 'VulnerabilityScan', 'Recon-PortScan', 'Recon-HostDiscovery', 'SqlInjection',
               'CommandInjection', 'Backdoor_Malware', 'Uploading_Attack', 'XSS', 'BrowserHijacking',
               'Mirai-greeth_flood', 'Mirai-greip_flood', 'Mirai-udpplain', 'BenignTraffic']

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Find the index of Benign Traffic class
benign_index = class_names.index('BenignTraffic')

# Calculate true positives for the Benign Traffic class
true_positives = np.sum((y_val[:, benign_index] == 1) & (y_val_pred[:, benign_index] == 1))

# If the confusion matrix size is smaller than the index, extend it
if benign_index >= conf_matrix.shape[0]:
    conf_matrix = np.pad(conf_matrix, ((0, benign_index - conf_matrix.shape[0] + 1), (0, benign_index - conf_matrix.shape[1] + 1)), mode='constant', constant_values=0)

# Update the correct entry in the confusion matrix
conf_matrix[benign_index, benign_index] = true_positives

# Set labels for the confusion matrix
class_names_with_benign = class_names.copy()

# Visualize the confusion matrix for all classes, including Benign Traffic
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names_with_benign, yticklabels=class_names_with_benign, cbar_kws={'label': 'Count'})

# Set labels and title
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for All Classes with Benign Traffic")

plt.show()

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for Benign Traffic
fpr, tpr, _ = roc_curve((y_val[:, benign_index] == 1).astype(int), y_val_pred[:, benign_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[benign_index]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Benign Traffic Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names_with_benign)),
    target_names=class_names_with_benign,
    output_dict=True
))


# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for each class
for i, class_label in enumerate(mlb.classes_):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr, tpr, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for Benign Traffic', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))


# In[ ]:





# In[ ]:


#######################
##$#$#$$#$#$#$
#$#$#$#$#$$$#$#$#$#$$#$$$$$$$$$$$$$$$$$$$$
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood','DictionaryBruteForce','DNS_Spoofing','MITM-ArpSpoofing','DoS-SYN_Flood'
'DoS-UDP_Flood','DoS-TCP_Flood','DoS-HTTP_Flood','Recon-PingSweep','Recon-OSScan',
'VulnerabilityScan','Recon-PortScan','Recon-HostDiscovery','SqlInjection','CommandInjection','Backdoor_Malware',
'Uploading_Attack','XSS','BrowserHijacking','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain','BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for each class
for i, class_label in enumerate(class_names):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_test[:, i] == 1).astype(int), y_test_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr, tpr, _ = roc_curve((y_test[:, 0] == 1).astype(int), y_test_pred[:, 0])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_names[0]}', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_test.argmax(axis=1),
    y_test_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood','DictionaryBruteForce','DNS_Spoofing','MITM-ArpSpoofing','DoS-SYN_Flood'
'DoS-UDP_Flood','DoS-TCP_Flood','DoS-HTTP_Flood','Recon-PingSweep','Recon-OSScan',
'VulnerabilityScan','Recon-PortScan','Recon-HostDiscovery','SqlInjection','CommandInjection','Backdoor_Malware',
'Uploading_Attack','XSS','BrowserHijacking','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain','BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for each class
for i, class_label in enumerate(class_names):
    # Check if there are positive samples for the class
    if np.sum(y_val[:, i]) > 0:
        # Compute ROC curve and area under the curve
        fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f}) for {class_label}')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Make sure 'y_val' and 'y_val_pred' are two-dimensional arrays
if y_val.ndim == 1:
    y_val = y_val.reshape(-1, 1)

if y_val_pred.ndim == 1:
    y_val_pred = y_val_pred.reshape(-1, 1)

# Replace 'class_names' with the actual list or array containing your class names
class_names = ['DDoS-ACK_Fragmentation','DDoS-UDP_Flood','DDoS-SlowLoris','DDoS-ICMP_Flood','DDoS-RSTFINFlood',
'DDoS-PSHACK_Flood','DDoS-HTTP_Flood','DDoS-UDP_Fragmentation','DDoS-ICMP_Fragmentation','DDoS-TCP_Flood','DDoS-SYN_Flood',
'DDoS-SynonymousIP_Flood','DictionaryBruteForce','DNS_Spoofing','MITM-ArpSpoofing','DoS-SYN_Flood'
'DoS-UDP_Flood','DoS-TCP_Flood','DoS-HTTP_Flood','Recon-PingSweep','Recon-OSScan',
'VulnerabilityScan','Recon-PortScan','Recon-HostDiscovery','SqlInjection','CommandInjection','Backdoor_Malware',
'Uploading_Attack','XSS','BrowserHijacking','Mirai-greeth_flood','Mirai-greip_flood','Mirai-udpplain','BenignTraffic']

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for each class
for i, class_label in enumerate(class_names):
    # Check if there are positive samples for the class
    if np.sum(y_val[:, i]) > 0:
        # Compute ROC curve and area under the curve
        fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f}) for {class_label}')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True
))


# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for each class
for i, class_label in enumerate(range(y_val.shape[1])):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for Class {class_label}')

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr, tpr, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for Benign Traffic', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(y_val.shape[1]),
    target_names=[f'Class {i}' for i in range(y_val.shape[1])],
    output_dict=True
))


# In[ ]:


###################&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#####################^^^^^^^^^^^^^^^^^^^^^^^^^^
###################^^^^^^^^^^^^^^^^^^^^^
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Flatten the ground truth and predicted values
y_test_flat = y_test.flatten()
y_test_pred_flat = y_test_pred.flatten()

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test_flat, y_test_pred_flat)

# Compute AUC for the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the overall ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'CICIoT 2023 (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CICIoT 2023 Dataset')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted probabilities)
# ...

# Flatten the ground truth labels
y_test_flat = y_test.ravel()

# Threshold the predicted probabilities to obtain binary predictions
threshold = 0.5  # Example threshold, you can adjust it based on your preference
y_test_pred_binary = (y_test_pred > threshold).astype(int)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_flat, y_test_pred_binary)

# Calculate metrics
accuracy = accuracy_score(y_test_flat, y_test_pred_binary)
precision = precision_score(y_test_flat, y_test_pred_binary)
recall = recall_score(y_test_flat, y_test_pred_binary)
f1 = f1_score(y_test_flat, y_test_pred_binary)

# Calculate true positive rate (TPR), true negative rate (TNR), false positive rate (FPR), false negative rate (FNR)
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

# Calculate Matthews correlation coefficient (MCC)
mcc = matthews_corrcoef(y_test_flat, y_test_pred_binary)

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test_flat, y_test_pred.ravel())

# Compute AUC for the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the overall ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'CICIoT 2023 (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CICIoT 2023 Dataset')
plt.legend(loc='lower right')
plt.show()

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'True Positive Rate (TPR): {TPR:.4f}')
print(f'True Negative Rate (TNR): {TNR:.4f}')
print(f'False Positive Rate (FPR): {FPR:.4f}')
print(f'False Negative Rate (FNR): {FNR:.4f}')
print(f'AUC: {roc_auc:.4f}')
print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')


# In[ ]:


##########
#######Aala 2
#########

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

# Assuming you have 'y_val_flat' (ground truth) and 'y_val_pred_flat' (predicted probabilities)
# Also assuming you have defined 'threshold' for binary classification

# Threshold the predicted probabilities to obtain binary predictions
threshold = 0.5  # Example threshold, you can adjust it based on your preference
y_test_pred_binary = (y_test_pred_flat > threshold).astype(int)

# Ensure that y_val_flat and y_val_pred_binary have the same number of samples
# Trim or pad y_val_pred_binary to match the number of samples in y_val_flat
if len(y_test_flat) != len(y_test_pred_binary):
    if len(y_test_flat) > len(y_test_pred_binary):
        y_test_flat = y_test_flat[:len(y_test_pred_binary)]
    else:
        y_test_pred_binary = y_test_pred_binary[:len(y_test_flat)]

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test_flat, y_test_pred_binary)

# Calculate metrics
accuracy = accuracy_score(y_test_flat, y_test_pred_binary)
precision = precision_score(y_test_flat, y_test_pred_binary)
recall = recall_score(y_test_flat, y_test_pred_binary)
f1 = f1_score(y_test_flat, y_test_pred_binary)

# Calculate true positive rate (TPR), true negative rate (TNR), false positive rate (FPR), false negative rate (FNR)
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

# Calculate Matthews correlation coefficient (MCC)
mcc = matthews_corrcoef(y_test_flat, y_test_pred_binary)

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_test_flat, y_test_pred_flat)

# Compute AUC for the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the overall ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'UNSW_NB15 (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('UNSW_NB15 Dataset')
plt.legend(loc='lower right')
plt.show()

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'True Positive Rate (TPR): {TPR:.4f}')
print(f'True Negative Rate (TNR): {TNR:.4f}')
print(f'False Positive Rate (FPR): {FPR:.4f}')
print(f'False Negative Rate (FNR): {FNR:.4f}')
print(f'AUC: {roc_auc:.4f}')
print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')


# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted probabilities)
# ...

# Threshold the predicted probabilities to obtain binary predictions
threshold = 0.5  # Example threshold, you can adjust it based on your preference
y_val_pred_binary = (y_val_pred > threshold).astype(int)

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_val_flat, y_val_pred_flat)

# Compute AUC for the ROC curve
roc_auc = auc(fpr, tpr)

# Compute other metrics
accuracy = accuracy_score(y_val_flat, y_val_pred_binary)
precision = precision_score(y_val_flat, y_val_pred_binary)
recall = recall_score(y_val_flat, y_val_pred_binary)
f1 = f1_score(y_val_flat, y_val_pred_binary)

conf_matrix = confusion_matrix(y_val_flat, y_val_pred_binary)
fpr = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])  # False Positive Rate
fnr = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # False Negative Rate
mcc = matthews_corrcoef(y_val_flat, y_val_pred_binary)


# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'False Positive Rate: {fpr:.4f}')
print(f'False Negative Rate: {fnr:.4f}')
print(f'Matthews Correlation Coefficient: {mcc:.4f}')

# Plot the overall ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'UNSW_NB15 (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('UNSW_NB15 Dataset')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


###############Aala
##################
##########################
#####################

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted probabilities)
# ...

# Threshold the predicted probabilities to obtain binary predictions
threshold = 0.5  # Example threshold, you can adjust it based on your preference
y_val_pred_binary = (y_val_pred > threshold).astype(int)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_val_flat, y_val_pred_binary)

# Calculate metrics
accuracy = accuracy_score(y_val_flat, y_val_pred_binary)
precision = precision_score(y_val_flat, y_val_pred_binary)
recall = recall_score(y_val_flat, y_val_pred_binary)
f1 = f1_score(y_val_flat, y_val_pred_binary)

# Calculate true positive rate (TPR), true negative rate (TNR), false positive rate (FPR), false negative rate (FNR)
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)

# Calculate Matthews correlation coefficient (MCC)
mcc = matthews_corrcoef(y_val_flat, y_val_pred_binary)

# Compute the ROC curve
fpr, tpr, _ = roc_curve(y_val_flat, y_val_pred_flat)

# Compute AUC for the ROC curve
roc_auc = auc(fpr, tpr)

# Plot the overall ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'CICIoT 2023 (AUC = {roc_auc:.4f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('CICIoT 2023 Dataset')
plt.legend(loc='lower right')
plt.show()

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'True Positive Rate (TPR): {TPR:.4f}')
print(f'True Negative Rate (TNR): {TNR:.4f}')
print(f'False Positive Rate (FPR): {FPR:.4f}')
print(f'False Negative Rate (FNR): {FNR:.4f}')
print(f'AUC: {roc_auc:.4f}')
print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')


# In[ ]:


# Check shapes of arrays
print("Shape of y_val_flat:", y_val_flat.shape)
print("Shape of y_val_pred_binary:", y_val_pred_binary.shape)

# Verify data generation process
# Ensure that y_val_flat and y_val_pred_binary are derived from the same dataset

# Debug prediction process if necessary
# Print intermediate variables, check loops, conditional statements, etc.

# Make necessary adjustments to align the number of samples
# Adjust data preprocessing steps or prediction code as needed


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Get the unique class labels from your data
unique_classes = np.unique(y_val)

# Create class names based on unique class labels
class_names = [f'Class {label}' for label in unique_classes]

# Flatten the ground truth and predicted values
y_val_flat = y_val.flatten()
y_val_pred_flat = y_val_pred.flatten()

# Calculate metrics
accuracy = accuracy_score(y_val_flat, y_val_pred_flat)
precision = precision_score(y_val_flat, y_val_pred_flat, average='weighted')
recall = recall_score(y_val_flat, y_val_pred_flat, average='weighted')
f1 = f1_score(y_val_flat, y_val_pred_flat, average='weighted')
mcc = matthews_corrcoef(y_val_flat, y_val_pred_flat)

# Convert metrics to percent format
accuracy_percent = accuracy * 100.0
precision_percent = precision * 100.0
recall_percent = recall * 100.0
f1_percent = f1 * 100.0
mcc_percent = mcc * 100.0

# Print the metrics
print(f"Accuracy: {accuracy_percent:.2f}%")
print(f"Precision: {precision_percent:.2f}%")
print(f"Recall: {recall_percent:.2f}%")
print(f"F1-Score: {f1_percent:.2f}%")
print(f"Matthews Correlation Coefficient: {mcc_percent:.2f}%")

# Print classification report
classification_rep = classification_report(y_val_flat, y_val_pred_flat, target_names=class_names)
print("Classification Report:\n", classification_rep)


# In[ ]:


from sklearn.metrics import classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Get the unique class labels from your data
unique_classes = np.unique(y_val)

# Create class names based on unique class labels
class_names = [f'Class {label}' for label in unique_classes]

# Print classification report
classification_rep = classification_report(y_val.flatten(), y_val_pred.flatten(), target_names=class_names)
print("Classification Report:\n", classification_rep)


# In[ ]:


from sklearn.metrics import classification_report

# Assuming you have 'y_val' (ground truth probabilities) and 'y_val_pred' (predicted probabilities)
# ...

# Convert probabilities to binary predictions (you can adjust the threshold as needed)
y_val_pred = model.predict(X_val)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)

# Flatten the ground truth and predicted values
y_val = y_val.flatten()
y_val_pred = y_val_pred_binary.flatten()

# Print classification report
classification_rep = classification_report(y_val, y_val_pred, target_names=class_names)
print("Classification Report:\n", classification_rep)


# In[ ]:


from sklearn.metrics import classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Flatten the ground truth and predicted values
y_val = y_val.flatten()
y_val_pred = y_val_pred.flatten()

# Print classification report
classification_rep = classification_report(y_val, y_val_pred, target_names=class_names)
print("Classification Report:\n", classification_rep)
#multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))


# In[ ]:


benign_class_index = 0

plt.figure(figsize=(15, 10))

for i in range(y_val.shape[1]):
    plt.subplot(6, 6, i + 1)  # Assuming you have 34 classes, adjust accordingly
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Class {i} vs. Benign (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class {i} vs. Benign')
    plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[ ]:


# Calculate metrics
accuracy = accuracy_score(y_val, y_val_pred,)
precision = precision_score(y_val, y_val_pred, average='weighted')
recall = recall_score(y_val, y_val_pred, average='weighted')
f1 = f1_score(y_val, y_val_pred, average='weighted')
mcc = matthews_corrcoef(y_val, y_val_pred,)
roc_auc = roc_auc_score(y_val, y_val_pred, multi_class='ovr')

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'Matthews Correlation Coefficient: {mcc:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')


# In[ ]:


benign_class_index = 0

plt.figure(figsize=(15, 40))  # Adjust figsize based on the number of classes

for i in range(y_val.shape[1]):
    plt.subplot(11, 4, i + 1)  # Assuming you have 34 classes, adjust accordingly
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'Class {i} vs. Benign (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class {i} vs. Benign')
    plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

benign_class_index = 0

plt.figure(figsize=(15, 40))  # Adjust figsize based on the number of classes

for i in range(y_val.shape[1]):
    if i != benign_class_index:
        fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
        roc_auc = auc(fpr, tpr)

        plt.subplot(11, 4, i + 1)  # Assuming you have 34 classes, adjust accordingly
        plt.plot(fpr, tpr, label=f'Class {i} vs. Benign (AUC = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {i} vs. Benign')
        plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Calculate and print accuracy, precision, recall, and ROC Curve for the entire dataset
y_val_pred = model.predict(X_val)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_val_pred_binary)
precision = precision_score(y_val, y_val_pred_binary, average='weighted')
recall = recall_score(y_val, y_val_pred_binary, average='weighted')
roc_auc = roc_auc_score(y_val, y_val_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_val.ravel(), y_val_pred.ravel())
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Entire Dataset')
plt.legend(loc="lower right")
plt.show()

# 2. Calculate multi-label confusion matrix
confusion_matrices = multilabel_confusion_matrix(y_val, y_val_pred_binary)

# 3. Print multi-label confusion matrix for each class with labels and heatmap
for i, conf_matrix in enumerate(confusion_matrices):
    print(f"\nConfusion Matrix - Class {i}:\n", conf_matrix)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Multi-label Confusion Matrix - Class {i}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# 4. Print overall classification report
classification_rep = classification_report(y_val, y_val_pred_binary, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n", classification_rep)

# 5. Sum the confusion matrices along the first axis to aggregate them
combined_confusion_matrix = np.sum(confusion_matrices, axis=0)

# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(combined_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Aggregated Multi-label Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 6. Plot ROC curve for each class with the Benign traffic class
benign_class_index = 0

plt.figure(figsize=(8, 6))
for i in range(y_val.shape[1]):
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    label = f'Class {i} (AUC = {roc_auc:.2f})'
    linestyle = '--' if i == benign_class_index else '-'
    plt.plot(fpr, tpr, label=label, linestyle=linestyle)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class with Benign Traffic Class')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns
# Make predictions on the test set
y_pred = model.predict(X_val)


# In[ ]:


# Convert predictions to binary (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)


# In[ ]:


# Flatten the ground truth and predictions for binary classification
y_val_flat = y_val.argmax(axis=1)
y_pred_flat = y_pred_binary.argmax(axis=1)


# In[ ]:


# Print accuracy, precision, recall
accuracy = accuracy_score(y_val_flat, y_pred_flat)
precision = precision_score(y_val_flat, y_pred_flat, average='macro')  # 'macro' calculates metrics for each label, and finds their unweighted mean
recall = recall_score(y_val_flat, y_pred_flat, average='macro')


# In[ ]:


#############################Overall
######################
##################
################
######
from sklearn.metrics import f1_score, matthews_corrcoef
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print("ROC AUC:", roc_auc)
#print(f'F1-score: {f1:.4f}')
print(f'Matthews Correlation Coefficient: {mcc:.4f}')
# Calculate F1-score
f1 = f1_score(y_true, y_pred)

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_true, y_pred)

# Print F1-score and Matthews Correlation Coefficient
print(f'F1-Score: {f1:.4f}')
print(f'Matthews Correlation Coefficient: {mcc:.4f}')


# In[ ]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef

# Assuming you have 'y_true' (ground truth) and 'y_pred' (predicted values)
# ...

# Calculate precision, recall, F1-score, ROC AUC, and Matthews correlation coefficient
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# Print the metrics
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print("ROC AUC:", roc_auc)
print(f'F1-score: {f1:.4f}')
print(f'Matthews Correlation Coefficient: {mcc:.4f}')


# In[ ]:


import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef

# Assuming you have 'y_true' (ground truth) and 'y_pred' (predicted values)
# ...

# Calculate precision, recall, F1-score, ROC AUC, and Matthews correlation coefficient
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# Print the evaluation metrics
print("Evaluation Metrics:")
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print("ROC AUC:", roc_auc)
print(f'F1-score: {f1:.4f}')
print(f'Matthews Correlation Coefficient: {mcc:.4f}')


# In[ ]:


# Plot ROC Curve for the entire dataset
fpr, tpr, _ = roc_curve(y_val_flat, y_pred_flat)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()


# In[ ]:


# Draw the confusion matrix for each label
confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)
for i, conf_matrix in enumerate(confusion_matrices):
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    plt.title(f'Confusion Matrix - Class {i}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[ ]:


# Draw the combined confusion matrix
combined_confusion_matrix = np.sum(confusion_matrices, axis=0)
sns.heatmap(combined_confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Combined Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:


# Plot ROC curves for each output class with the benign traffic class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class with the benign traffic class
plt.figure(figsize=(8, 6))
plt.plot(fpr[0], tpr[0], label=f'Class 0 (Benign) - AUC = {roc_auc[0]:.2f}', linestyle='--')
for i in range(1, y_test.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} - AUC = {roc_auc[i]:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class with Benign Traffic Class')
plt.legend()
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns

# Make predictions on the test set
y_pred = model.predict(X_val_filtered)

# Convert predictions to binary (0 or 1)
y_pred_binary = (y_pred > 0.5).astype(int)

# Flatten the ground truth and predictions for binary classification
y_val_flat = y_val.argmax(axis=1)
y_pred_flat = y_pred_binary.argmax(axis=1)

# Print accuracy, precision, recall
accuracy = accuracy_score(y_val_flat, y_pred_flat)
precision = precision_score(y_val_flat, y_pred_flat, average='macro')  # 'macro' calculates metrics for each label, and finds their unweighted mean
recall = recall_score(y_val_flat, y_pred_flat, average='macro')
roc_auc = roc_auc_score(y_val_flat, y_pred_flat, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print("ROC AUC:", roc_auc)
# Plot ROC Curve for the entire dataset
fpr, tpr, _ = roc_curve(y_val_flat, y_pred_flat)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()

# Draw confusion matrix for each label
class_names = ['Class 0', 'Class 1', 'Class 2', ...]  # Replace with your class names
cm_labels = [f'True {class_name}' for class_name in class_names]
cm = confusion_matrix(y_val_flat, y_pred_flat)

# Plot combined confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Combined Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot confusion matrix for each label
for i in range(len(class_names)):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix - {class_names[i]}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# ROC curve of each output class with the benign traffic class
roc_auc_classes = []
for i in range(y_pred.shape[1]):
    if i == 0:  # Assuming the first class is benign
        continue
    fpr, tpr, _ = roc_curve(y_val[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_classes.append(roc_auc)

plt.figure(figsize=(8, 6))
for i in range(1, len(class_names)):
    plt.plot(fpr, tpr, label=f'Class {i} vs. Benign (AUC = {roc_auc_classes[i-1]:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Output Class with Benign Traffic Class')
plt.legend()
plt.show()


# In[ ]:


_,acc = model.evaluate(X_test, y_test)
print("Accuracy =", (acc * 100.0), "%")
_,pre = model.evaluate(X_test, y_test)
print("Precision =", (pre * 100.0), "%")
_,f1score = model.evaluate(X_test, y_test)
print("F1 Score =", (f1score * 100.0), "%")
_,rec = model.evaluate(X_test, y_test)
print("Recall =", (rec * 100.0), "%")
_,score = model.evaluate(X_test, y_test)
print('Test score:', (score * 100.0), "%")


# In[ ]:


# Evaluate the model on training set
train_loss, train_accuracy = model.evaluate(X_train, y_train)


# In[ ]:





# In[ ]:


# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)


# In[ ]:


# Make predictions on validation set
y_val_pred = model.predict(X_val)


# In[ ]:


# Calculate multilabel confusion matrix
conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))


# In[ ]:


_,acc = model.evaluate(X_test, y_test)
print("Accuracy =", (acc * 100.0), "%")
_,pre = model.evaluate(X_test, y_test)
print("Precision =", (pre * 100.0), "%")
_,f1score = model.evaluate(X_test, y_test)
print("F1 Score =", (f1score * 100.0), "%")
_,rec = model.evaluate(X_test, y_test)
print("Recall =", (rec * 100.0), "%")
_,score = model.evaluate(X_test, y_test)
print('Test score:', (score * 100.0), "%")


# In[ ]:


# Sum the confusion matrices along the first axis to aggregate them
agg_cm = np.sum(conf_matrix, axis=0)

# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Aggregated Confusion Matrix')
plt.show()


# In[ ]:


# Calculate precision, recall, f1-score, and Matthews correlation coefficient
#classification_rep = classification_report(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), target_names=mlb.classes_, output_dict=True)
classification_rep = classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
)

classification_rep = classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
)
print(classification_rep)
precision = classification_rep['weighted avg']['precision']
print(precision)
recall = classification_rep['weighted avg']['recall']
print(recall)
f1_score = classification_rep['weighted avg']['f1-score']
print(f1_score)
mcc = matthews_corrcoef(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))
print(matthews_corrcoef)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=0), y_val_pred.argmax(axis=0))

# Print multi-label confusion matrix for each class with labels and heatmap
for i, class_label in enumerate(mlb.classes_):
    print(f"\nConfusion Matrix for Class '{class_label}':")
    print(multilabel_conf_matrix[i])

    # Plot heatmap for the current class
    plt.figure(figsize=(8, 6))
    #sns.heatmap(multilabel_conf_matrix[i], annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    sns.heatmap(multilabel_conf_matrix[i], annot=True,  cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Confusion Matrix for Class \'{class_label}\'')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=0),
    y_val_pred.argmax(axis=0),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))


# In[ ]:


# Sum the confusion matrices along the first axis to aggregate them
agg_cm = np.sum(multilabel_conf_matrix, axis=0)

# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(agg_cm, annot=True, cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Aggregated Confusion Matrix')
plt.show()


# In[ ]:


print("Size of multilabel_conf_matrix:", len(multilabel_conf_matrix))
for i, class_label in enumerate(mlb.classes_):
    if i <= len(multilabel_conf_matrix):
        print(f"\nConfusion Matrix for Class '{class_label}':")
        print(multilabel_conf_matrix[i])
        # Rest of your code
    else:
        print(f"Index {i} is out of bounds for the array.")
print("Number of classes:", len(mlb.classes_))


# In[ ]:


# Print the results
print("Multilabel Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))
#print(classification_report(y_val.argmax(axis=1), y_val_pred.argmax(axis=1), target_names=mlb.classes_))
print("\nTraining Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)
print("Matthews Correlation Coefficient:", mcc)


# In[ ]:


# Plot ROC curve for each class
plt.figure(figsize=(10, 7))
for i in range(y_val.shape[1]):
    fpr, tpr, _ = roc_curve(y_val[:, i], y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Initialize a figure for ROC curves
plt.figure(figsize=(10, 8))

# Plot ROC curve for each class
for i, class_label in enumerate(mlb.classes_):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')

# Plot ROC curve for Benign Traffic (assuming it's the first class)
fpr, tpr, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for Benign Traffic', linestyle='--', color='black')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Plot ROC curve for each class
for i, class_label in enumerate(mlb.classes_):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}')
    
    # Plot ROC curve for Benign Traffic (assuming it's the first class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic', linestyle='--', color='green')
# Plot the diagonal line for reference
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))
#plt.plot(fpr[i], tpr[i], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f})')
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Calculate multi-label confusion matrix
multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))

# Define a color map for the ROC curves
colors = plt.cm.get_cmap('tab10', len(mlb.classes_)+1)

# Plot ROC curve for each class
for i, class_label in enumerate(mlb.classes_):
    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve((y_val[:, i] == 1).astype(int), y_val_pred[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the current class with a unique color
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {class_label}', color=colors(i))
    
    # Plot diagonal line for random guessing
    #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
     # Plot diagonal line for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # Plot ROC curve for Benign Traffic (assuming it's the first class)
    fpr_benign, tpr_benign, _ = roc_curve((y_val[:, 0] == 1).astype(int), y_val_pred[:, 0])
    roc_auc_benign = auc(fpr_benign, tpr_benign)
    plt.plot(fpr_benign, tpr_benign, label=f'ROC curve (area = {roc_auc_benign:.2f}) for Benign Traffic', linestyle='--', color='black')

    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class \'{class_label}\' with Benign Traffic')
    plt.legend(loc='lower right')
    plt.show()

# Print overall classification report
print("\nOverall Classification Report:")
print(classification_report(
    y_val.argmax(axis=1),
    y_val_pred.argmax(axis=1),
    labels=np.arange(len(mlb.classes_)),
    target_names=mlb.classes_,
    output_dict=True
))


# In[ ]:


# Sum the confusion matrices along the first axis to aggregate them
agg_cm = np.sum(multilabel_conf_matrix, axis=0)

# Plot the aggregated heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Aggregated Confusion Matrix')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


# Evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics
multilabel_cm = multilabel_confusion_matrix(y_test, (y_pred > 0.5).astype(int))
classification_rep = classification_report(y_test, (y_pred > 0.5).astype(int), target_names=multilabel_cm.classes_)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:


print(X_test.size)
# X_train_filtered already has the shape (140035, 12, 1) for CNN
X_train = X_train_filtered
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Adjust shape based on X_test.size
#X_val = np.reshape(X_val_filtered, (X_val_filtered.shape[0], X_val_filtered.shape[1], 1))
X_val = X_val_filtered


# In[ ]:


# Reshape the data for the CNN
#X_train = np.reshape(X_train_filtered, (X_train_filtered.shape[0], X_train_filtered.shape[1],1))
#X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1],1))
#X_val = np.reshape(X_val, (X_val_filtered.shape[0], X_val_filtered.shape[1],1))


# In[ ]:


# Build a simple CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
#model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
          


# In[ ]:


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train, 
                    validation_data=(X_val.reshape((X_val.shape[0], X_val.shape[1], 1)), y_val),
                    epochs=5, batch_size=64, verbose=1)


# In[ ]:


_,acc = model.evaluate(X_test, y_test)
print("Accuracy =", (acc * 100.0), "%")
_,pre = model.evaluate(X_test, y_test)
print("Precision =", (pre * 100.0), "%")
_,f1score = model.evaluate(X_test, y_test)
print("F1 Score =", (f1score * 100.0), "%")
_,rec = model.evaluate(X_test, y_test)
print("Recall =", (rec * 100.0), "%")
_,score = model.evaluate(X_test, y_test)
print('Test score:', (score * 100.0), "%")


# In[ ]:


# Print other metrics
print('Precision: %.4f' % precision_score(y_test_2d, y_pred_2d, average='micro'))
print('Recall: %.4f' % recall_score(y_test_2d, y_pred_2d, average='micro'))
print('Accuracy: %.4f' % accuracy_score(y_test_2d, y_pred_2d))
print('F1 Score: %.4f' % f1_score(y_test_2d, y_pred_2d, average='micro'))
print('Jaccard Score: %.4f' % jaccard_score(y_test_2d, y_pred_2d, average='micro'))
print('Log Loss: %.4f' % log_loss(y_test_2d, y_pred_2d))
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test_2d, y_pred_2d))
print("ROC AUC Score:")
print(roc_auc_score(y_test_2d, y_pred_2d))


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

# Example class labels
#class_labels = model.classes_

class_labels = ['Class 0', 'Class 1', 'Class 2','class 3','class4','class 5','class 6', 'class 7', 'class 8', 
                'class 9','class 10','class 11','class 12','class 13', 'class 14','class 15','class 16',
              'class 17','class 18','class 19','class 20','class 21','class 22','class 23','class 24', 
                'class 25','class 26','class 27','class 28','class 29','class 30','class 31','class 32','class 33']

# Assuming you have a multilabel confusion matrix named 'cm'
# The shape of 'cm' should be (n_classes, 2, 2)

# Iterate over each label and plot a heatmap
for i in range(cm.shape[0]):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', xticklabels=[f'Predicted {class_labels[i]}', f'Predicted Not {class_labels[i]}'],
                yticklabels=[f'Actual {class_labels[i]}', f'Actual Not {class_labels[i]}'])
    plt.title(f'Confusion Matrix - {class_labels[i]}')
    plt.show()


# In[ ]:


# Print and plot metrics
def print_and_plot_metrics(y_true, y_pred, prefix):
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate multilabel confusion matrix
    cm = multilabel_confusion_matrix(y_true, y_pred_binary)

    # Print metrics
    print(f"\n{prefix} Metrics:")
    for i in range(len(mlb.classes_)):
        print(f"Class {mlb.classes_[i]}:")
        print(f"Confusion Matrix:\n{cm[i]}")
        print(f"Precision: {precision_score(y_true[:, i], y_pred_binary[:, i])}")
        print(f"Recall: {recall_score(y_true[:, i], y_pred_binary[:, i])}")
        print(f"F1-Score: {f1_score(y_true[:, i], y_pred_binary[:, i])}")
        print(f"AUC-ROC: {roc_auc_score(y_true[:, i], y_pred[:, i])}\n")

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    for i in range(len(mlb.classes_)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'Class {mlb.classes_[i]}')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} ROC Curve')
    plt.legend()
    plt.show()

print_and_plot_metrics(y_train, y_train_pred, prefix="Training")
print_and_plot_metrics(y_val, y_val_pred, prefix="Validation")


# In[ ]:


# Evaluate the model
#y_pred = model.predict(X_test_pca.reshape((X_test_pca.shape[0], X_test_pca.shape[1], 1)))
#y_pred = model.predict_classes(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)))
# Assuming y_test and y_pred are your true and predicted multilabel targets
# Reshape y_test and y_pred to 2D
# Now reshape y_pred

y_test = y_test.reshape(-1, y_test.shape[-1])
y_pred = y_pred.reshape(-1, y_pred.shape[-1])

# Calculate multilabel confusion matrix
cm = multilabel_confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Print other metrics
print('Precision: %.4f' % precision_score(y_test, y_pred, average='micro'))
print('Recall: %.4f' % recall_score(y_test, y_pred, average='micro'))
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.4f' % f1_score(y_test, y_pred, average='micro'))
print('Jaccard Score: %.4f' % jaccard_score(y_test, y_pred, average='micro'))
print('Log Loss: %.4f' % log_loss(y_test, y_pred))
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred))



# In[ ]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
cm =  multilabel_confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#print("Confusion Matrix:")
print('Precision: %.4f' % precision_score(y_test, y_pred, average='micro'))
print('Recall: %.4f' % recall_score(y_test, y_pred, average='micro'))
print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.4f' % f1_score(y_test, y_pred))
print('Jaccard Score: %.4f' % jaccard_score(y_test, y_pred))
print('Log Loss: %.4f' % log_loss(y_test, y_pred))
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC AUC Score:")
print(roc_auc_score(y_test, y_pred))


# In[ ]:


#Using Pearson Correlation
#cor = np.corrcoef(X_train, rowvar=False)
#plt.figure(figsize=(12,10))
##cor = X_train.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
#plt.show()


# In[ ]:


#def correlation (dataset, threshold):
#   corrcoef_matrix = dataset.corrcoef()
 #   for i in range (len(corrcoef_matrix.columns)):
  #      for j in range(i):
   #         if abs (corrcoef_matrix.iloc[i,j]) > threshold:
    #            colname= corrcoef_matrix.columns[i]
     #           col_corrcoef.add(colname)
      #          return col_corrcoef


# In[ ]:


#corrcoef_features = correlation(X_train, 0.8)
#len(set(corrcoef_features))


# In[ ]:


X_train.var(axis=0)


# In[ ]:


fig, ax = plt.subplots()
x = X.columns
y = X_train.var(axis=0)

ax.bar(x,y, width=0.2)
ax.set_xlabel('Features')
ax.set_ylabel('Variance')
ax.set_ylim(0,0.1)

for index, value in enumerate(y):
    plt.text(x=index, y=value+0.001, s=str(round(value, 3)), ha='center')
    
    fig.autofmt_xdate()
    plt.tight_layout()


# In[ ]:





# In[ ]:





# In[ ]:


# Define XGBOOST classifier to be used by Boruta
import xgboost as xgb
model = xgb.XGBClassifier()  #For Boruta


# In[ ]:


#Create shadow features – random features and shuffle values in columns
#Train Random Forest / XGBoost and calculate feature importance via mean decrease impurity
#Check if real features have higher importance compared to shadow features 
#Repeat this for every iteration
#If original feature performed better, then mark it as important 
#"""

from boruta import BorutaPy


# In[ ]:


# define Boruta feature selection method
feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)


# In[ ]:


# find all relevant features
feat_selector.fit(X_train, y_train)


# In[ ]:


# check selected features
print(feat_selector.support_)  #Should we accept the feature


# In[ ]:


# check ranking of features
print(feat_selector.ranking_) #Rank 1 is the best


# In[ ]:


# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X_train)  #Apply feature selection and return transformed data


# In[ ]:


#Review the features
#"""
# zip feature names, ranks, and decisions 
feature_ranks = list(zip(feature_names, 
                         feat_selector.ranking_, 
                         feat_selector.support_))


# In[ ]:


# print the results
for feat in feature_ranks:
    print('Feature: {:<30} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))


# In[ ]:


#Now use the subset of features to fit XGBoost model on training data
import xgboost as xgb
xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_filtered, y_train)


# In[ ]:


#Now predict on test data using the trained model. 

#First apply feature selector transform to make sure same features are selected from test data
X_test_filtered = feat_selector.transform(X_test)
prediction_xgb = xgb_model.predict(X_test_filtered)


# In[ ]:


#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_xgb))


# In[ ]:


#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_xgb)
#print(cm)
sns.heatmap(cm, annot=True)


# In[ ]:


label

DDoS-ACK_Fragmentation
DDoS-UDP_Flood
DDoS-SlowLoris
DDoS-ICMP_Flood
DDoS-RSTFINFlood
DDoS-PSHACK_Flood
DDoS-HTTP_Flood
DDoS-UDP_Fragmentation
DDoS-ICMP_Fragmentation
DDoS-TCP_Flood
DDoS-SYN_Flood
DDoS-SynonymousIP_Flood

DictionaryBruteForce

DNS_Spoofing
MITM-ArpSpoofing

DoS-SYN_Flood
DoS-UDP_Flood
DoS-TCP_Flood
DoS-HTTP_Flood

Recon-PingSweep
Recon-OSScan
VulnerabilityScan
Recon-PortScan
Recon-HostDiscovery

SqlInjection
CommandInjection
Backdoor_Malware
Uploading_Attack
XSS
BrowserHijacking


Mirai-greeth_flood
Mirai-greip_flood
Mirai-udpplain
BenignTraffic



# In[ ]:


How to resolve  error in this code
# Assuming X_train, X_val are DataFrames
X_train_array = X_train.values
X_test_array = X_test.values
X_val_array = X_val

# Reshape the data for the CNN
X_train_reshaped = np.reshape(X_train_array, (X_train_array.shape[0], X_train_array.shape[1], 1))
X_test_reshaped = np.reshape(X_test_array, (X_test_array.shape[0], X_test_array.shape[1], 1))
X_val_reshaped = np.reshape(X_val_array, (X_val_array.shape[0], X_val_array.shape[1], 1))
# Build a simple CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_array.shape[1], 1)))
#model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[24], line 2
      1 # Train the model
----> 2 history = model.fit(X_train.reshape((X_train_array.shape[0], X_train_array.shape[1], 1)), y_train, 
      3                     validation_data=(X_val_array.reshape((X_val_array.shape[0], X_val_array.shape[1], 1)), y_val),
      4                     epochs=100, batch_size=64, verbose=1)

File C:\ProgramData\anaconda3\lib\site-packages\pandas\core\generic.py:5902, in NDFrame.__getattr__(self, name)
   5895 if (
   5896     name not in self._internal_names_set
   5897     and name not in self._metadata
   5898     and name not in self._accessors
   5899     and self._info_axis._can_hold_identifiers_and_holds_name(name)
   5900 ):
   5901     return self[name]
-> 5902 return object.__getattribute__(self, name)

AttributeError: 'DataFrame' object has no attribute 'reshape'


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, multilabel_confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import baruta
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, auc

# Load the dataset
data = pd.read_csv('data.csv')

# Separate features (X) and labels (y)
X = data.iloc[:, :-1].values  # Assuming the independent variables are in the first 45 columns
y = data.iloc[:, -1].values  # Assuming the dependent variable is in the last column

# One-hot encode the dependent variable
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Feature scaling for independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Feature selection using Baruta
selector = baruta.Baruta()
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)

# Drop highly correlated features
correlation_matrix = pd.DataFrame(X_train_selected).corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
X_train_selected = X_train_selected.drop(to_drop, axis=1)
X_val_selected = X_val_selected.drop(to_drop, axis=1)

# Use SMOTE to address data imbalance
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_selected, y_train)

# Build CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_selected.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train_balanced.shape[1], activation='sigmoid'))  # Assuming a binary classification for each label

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_balanced.reshape(X_train_balanced.shape[0], X_train_balanced.shape[1], 1),
    y_train_balanced,
    epochs=50,
    validation_data=(X_val_selected.reshape(X_val_selected.shape[0], X_val_selected.shape[1], 1), y_val),
    batch_size=32
)

# Evaluate the model on test set
test_loss, test_accuracy = model.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Predict on the test set
y_pred = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))

# Convert predictions to binary
y_pred_binary = (y_pred > 0.5).astype(int)

# Print classification report
print(classification_report(y_test, y_pred_binary))

# Calculate ROC AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
for i in range(y_test.shape[1]):
    plt.figure()
    plt.plot(fpr[i], tpr[i], color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class {i}')
    plt.legend(loc='lower right')
    plt.show()

# Calculate multilabel confusion matrix
conf_matrix_multilabel = multilabel_confusion_matrix(y_test, y_pred_binary)

# Print confusion matrix for each class
for i in range(y_test.shape[1]):
    print(f"Confusion Matrix for Class {i}:\n", conf_matrix_multilabel[i])

# Calculate true positive rate, false positive rate, false negative rate, and true negative rate
tpr = conf_matrix_multilabel[:, 1, 1] / (conf_matrix_multilabel[:, 1, 1] + conf_matrix_multilabel[:, 1, 0])
fpr = conf_matrix_multilabel[:, 0, 1] / (conf_matrix_multilabel[:, 0, 1] + conf_matrix_multilabel[:, 0, 0])
fnr = conf_matrix_multilabel[:, 1, 0] / (conf_matrix_multilabel[:, 1, 0] + conf_matrix_multilabel[:, 1, 1])
tnr = conf_matrix_multilabel[:, 0, 0] / (conf_matrix_multilabel[:, 0, 0] + conf_matrix_multilabel[:, 0, 1])

# Plot ROC curve based on training and validation errors
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Error')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Assuming X_train and y_train are your training data
# Assuming X_test and y_test are your test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X_train and X_test if needed
X_train = X_train.values.reshape((X_train.shape[0], -1))
X_test = X_test.values.reshape((X_test.shape[0], -1))

# Create and fit the Logistic Regression model
classifier = LogisticRegression(max_iter=1000)  # You can adjust max_iter based on convergence
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming X_train and y_train are your training data
# Check the shape of X_train and y_train
print(X_train.shape, y_train.shape)

# Reshape X_train if needed
X_train = X_train.reshape((X_train.shape[0], -1))

# Create and fit the Gaussian Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Assuming X_test and y_test are your test data
# Reshape X_test if needed
X_test = X_test.reshape((X_test.shape[0], -1))

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)



# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Assuming X_train and y_train are your training data
# Assuming X_test and y_test are your test data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Reshape X_train and X_test if needed
X_train = X_train.values.reshape((X_train.shape[0], -1))
X_test = X_test.values.reshape((X_test.shape[0], -1))

# Create and fit the KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming X_train and y_train are your training data
# Check the shape of X_train and y_train
print(X_train.shape, y_train.shape)

# Reshape X_train if needed
X_train = X_train.reshape((X_train.shape[0], -1))

# Create and fit the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Assuming X_test and y_test are your test data
# Reshape X_test if needed
X_test = X_test.reshape((X_test.shape[0], -1))

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data.csv')

# Separate independent and dependent variables
X = data.iloc[:, :-1]  # Assuming the independent variables are in the first 45 columns
y = data.iloc[:, -1]   # Assuming the dependent variable is in the last column

# Perform one-hot encoding for the dependent variable
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()

# Feature scaling for independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE for class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# Split the dataset into training, testing, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Select features based on correl

# Train and evaluate various models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    
    precision = precision_score(y_val, y_pred_val, average='weighted')
    recall = recall_score(y_val, y_pred_val, average='weighted')
    accuracy = accuracy_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val, average='weighted')
    mcc = matthews_corrcoef(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))
    
    results[name] = {
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "F1-score": f1,
        "MCC": mcc
    }
    
    # Plot ROC curve
    if name == "Logistic Regression" or name == "Random Forest":
        y_pred_prob_val = model.predict_proba(X_val)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(y_val.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_pred_prob_val[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(y_val.shape[1]):
            plt.plot(fpr[i], tpr[i], label='ROC curve (class {})'.format(i))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for {}'.format(name))
        plt.legend()
        plt.show()

# Display results
for name, result in results.items():
    print("Model:", name)
    print("Precision:", result["Precision"])
    print("Recall:", result["Recall"])
    print("Accuracy:", result["Accuracy"])
    print("F1-score:", result["F1-score"])
    print("MCC:", result["MCC"])
    print("\n")


# In[ ]:


from sklearn.metrics import classification_report

# Assuming you have 'y_val' (ground truth probabilities) and 'y_val_pred' (predicted probabilities)
# ...

# Convert probabilities to binary predictions (you can adjust the threshold as needed)
y_val_pred = model.predict(X_val_filtered)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)

# Flatten the ground truth and predicted values
y_val = y_val.flatten()
y_val_pred = y_val_pred_binary.flatten()


# Print classification report
classification_rep = classification_report(y_val, y_val_pred, target_names=class_names)
print("Classification Report:\n", classification_rep)


# In[ ]:


from sklearn.metrics import classification_report

# Assuming you have 'y_val' (ground truth) and 'y_val_pred' (predicted values)
# ...

# Flatten the ground truth and predicted values
y_val = y_val.flatten()
y_val_pred = y_val_pred.flatten()

# Print classification report
classification_rep = classification_report(y_val, y_val_pred, target_names=class_names)
print("Classification Report:\n", classification_rep)
#multilabel_conf_matrix = multilabel_confusion_matrix(y_val.argmax(axis=1), y_val_pred.argmax(axis=1))


# In[ ]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

# Assuming you have 'y_val' (ground truth probabilities) and 'y_val_pred' (predicted probabilities)
# ...

# Convert probabilities to binary predictions (you can adjust the threshold as needed)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)

# Flatten the ground truth and predicted values
y_val = y_val.flatten()
y_val_pred = y_val_pred_binary.flatten()

# Calculate metrics
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
mcc = matthews_corrcoef(y_val, y_val_pred)

# Print the metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {roc_auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

X = data.iloc[:, :-1].values # Independent variables
y = data.iloc[:, -1].values # Dependent variable

# Assuming X is a numpy array
X = pd.DataFrame(X)  # Convert X to a pandas DataFrame

# Now you can use X with ColumnTransformer
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Define numeric and categorical features
numeric_features = list(numeric_columns)
categorical_features = list(categorical_columns)

# Define preprocessing steps for numeric and categorical features
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Use mean imputation for numeric features
    ('scaler', StandardScaler())  # Standardize numeric features
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Use most frequent imputation for categorical features
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine numeric and categorical preprocessing pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Apply preprocessing pipeline to the entire dataset
X_preprocessed = preprocessor.fit_transform(X)

# Perform one-hot encoding for the dependent variable
#encoder = OneHotEncoder()
#y_encoded = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
# Perform one-hot encoding for the dependent variable
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()

# Apply SMOTE oversampling
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y_encoded)

# Split the dataset into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.9, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,train_size=0.9, test_size=0.1, random_state=42)

