import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

 
# AUC for a binary classifier
def auc(y_true, y_pred):
    # PFA, prob false alert for binary classifier
    def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # N = total number of negative labels
        N = K.sum(1 - y_true)
        # FP = total number of false alerts, alerts from the negative class labels
        FP = K.sum(y_pred - y_pred * y_true)
        return FP/N

    # P_TA prob true alerts for binary classifier
    def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        # P = total number of positive labels
        P = K.sum(y_true)
        # TP = total number of correct alerts, alerts from the positive class labels
        TP = K.sum(y_pred * y_true)
        return TP/P

    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    
    return K.sum(s, axis=0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------


def f1(y_true, y_pred):
    # 计算recall
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    # 计算precision
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#-----------------------------------------------------------------------------------------------------------------------------------------------------





#################   TEST   #################
# 构建
input1 = tf.keras.Input(shape=[x_train.shape[0],],dtype=float32)
X1 = tf.keras.layers.Flatten()(input1)
X1 = tf.keras.layers.BatchNormalization()(X1)
X1 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(X1)
X1 = tf.keras.layers.Dropout(0.5)(X1)
output = tf.keras.layers.Dense(1, activation='sigmoid')(X1)
 
model = tf.keras.Model(inputs=input1,outputs=output)
 
model.sammary()
 
# 编译
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc]
)