def learn(self, data, annotations, data_filenames=None):
    """Learn based on data and annotations

    Parameters
    ----------
    data : dict of FeatureContainers
        Feature data
    annotations : dict of MetadataContainers
        Meta data
    data_filenames : dict of filenames
        Filenames of stored data

    Returns
    -------
    self

    """

    #from keras.models import Sequential, Model
    '''
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D
    from keras.layers import Dense, Dropout, Activation, Flatten, Input, Dense, LSTM, Lambda, Embedding, Reshape
    from keras.layers.merge import Concatenate
    from keras.layers import BatchNormalization
    from keras.optimizers import Adam
    from keras.layers import LSTM
    from keras import backend as K
    import tensorflow as tf
    import numpy as np
    '''
    path = '/data/tmpsrt1/DCASE2017-baseline-system/applications/'
    num_epoch = 30
    batch_size = 256
    num_feature = 200
    num_label = 15
    dim_vector = 256
    margin = 0.5
    k_size = 256
    '''

    X_training = self.prepare_data(data=data, files=training_files)


    Y_training = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=training_files)

    if self.show_extra_debug:
        self.logger.debug('  Training items \t[{examples:d}]'.format(examples=len(X_training)))

    # Process validation data
    if validation_files:
        X_validation = self.prepare_data(data=data, files=validation_files)
        Y_validation = self.prepare_activity(activity_matrix_dict=activity_matrix_dict, files=validation_files)

        validation = (X_validation, Y_validation)
        if self.show_extra_debug:
            self.logger.debug('  Validation items \t[{validation:d}]'.format(validation=len(X_validation)))
    else:
        validation = None
    '''
    # Set seed
    self.set_seed()

    # Setup Keras
    self._setup_keras()

    with SuppressStdoutAndStderr():
        # Import keras and suppress backend announcement printed to stderr
        import keras


    print("@#@$@@ Building !@#@#!!")

    # Create model
    self.create_model()

    print("@#@$@@ Builded !@#@#!!")

    if self.show_extra_debug:
        self.log_model_summary()

    # Create callbacks
    callback_list = self.create_callback_list()
    # Set seed
    self.set_seed()

    #X_training = X_training.reshape(X_training.shape[0], X_training.shape[1], 1)
    print("asdasd")

    print(X_training.shape)
    #print(X_1.shape)
    #print(X_1)
    print(Y_training.shape)

    print("asdasd")
    from keras.callbacks import ModelCheckpoint
    #Train(X_training, Y_training, validation)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
    checkpointer = ModelCheckpoint(filepath='/data/tmpsrt1/log_new/wph', save_best_only=True, period = 10, verbose = 1,save_weights_only = True)
    self.model.fit(
        x = X_training,
        y = Y_training,
        batch_size=64,
        epochs=10000,#self.learner_params.get_path('training.epochs', 1),
        validation_data=validation,
        #validation_split = 0.1,
        verbose=2,
        shuffle=True,
        callbacks=[tbCallBack,checkpointer]
    )
    

    '''
    # Manually update callbacks
    for callback in callback_list:
        if hasattr(callback, 'close'):
            callback.close()

    for callback in callback_list:
        if isinstance(callback, StasherCallback):
            callback.log()
            best_weights = callback.get_best()['weights']
            if best_weights:
                self.model.set_weights(best_weights)
            break

    self['learning_history'] = hist.history
    '''

def _frame_probabilities(self, feature_data):

    return self.model.predict(x=feature_data).T

def create_model(self, input_shape):
    """Create sequential Keras model
    """


    from keras.models import Sequential, Model
    from keras.layers.advanced_activations import LeakyReLU, PReLU
    from keras.layers import Dense, Dropout, Flatten,Permute
    from keras.layers import Conv1D, MaxPooling1D,Conv2D,MaxPooling2D, AveragePooling2D
    from keras.layers import Dense, Dropout, Activation, Flatten, Input, Dense, LSTM, Lambda, Embedding, Reshape
    from keras.layers.merge import Concatenate
    from keras.layers import BatchNormalization
    from keras.optimizers import Adam
    from keras.constraints import max_norm
    from keras.layers import LSTM,Add
    from keras import backend as K
    import tensorflow as tf
    import numpy as np
    from .keras_STFT_layer_master.stft import Spectrogram
    from .keras_STFT_layer_master.melgram import Melspectrogram

    

    '''
    self.model = Sequential()
<<<<<<< HEAD

    

    self.model.add(Conv1D(256, 3, activation='relu', input_shape=(1764,1)))
    self.model.add(Dropout(0.25))
=======
    self.model.add(Conv1D(256, 3, activation='relu', input_shape=(1764/2,1)))
    #self.model.add(Dropout(0.25))
>>>>>>> origin/master
    #self.model.add(BatchNormalization())
    self.model.add(Conv1D(512, 3, activation='relu'))
    #self.model.add(BatchNormalization())
    self.model.add(MaxPooling1D(pool_size=2))
    self.model.add(Conv1D(512, 3, activation='relu'))
    self.model.add(Conv1D(512, 3, activation='relu'))
    self.model.add(MaxPooling1D(pool_size=2))
    self.model.add(Dropout(0.25))
    self.model.add(Flatten())
    self.model.add(Dense(15, activation='softmax'))
    #self.model.add(Conv1D(64, 3, activation='relu'))
    #self.model.add(BatchNormalization())
    #self.model.add(Conv1D(64, 3, activation='relu'))
    #self.model.add(BatchNormalization())
    #self.model.add(MaxPooling1D(pool_size=2))
    #self.model.add(Dropout(0.25))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    '''


    '''
    a = np.array(dim_vector)

    def hinge(A, B, C):
        return K.mean(K.maximum(0.0, margin + K.sum(tf.multiply(A, C), axis=-1) - K.sum(tf.multiply(A, B), axis=-1)))

    def Cos_is(x, v):
        return K.mean(K.maximum(0.0, 1 - margin - K.sum(tf.multiply(x, v), axis=-1)))

    def hinge_loss(A, B, C):
        return K.mean(K.maximum(margin
                         + K.transpose(K.transpose(tf.matmul(A, K.transpose(C))) - 
                            K.sum(tf.multiply(A, B), axis = -1)),
                         0.0),
                         axis=-1, keepdims=True)

    def hinge_1(A, B):
        return hinge_loss(A, B, B)

    def Loss1(y_true, y_pred):
        l_i, l_k, f_i, f_k = y_pred[:, a * 0 : a], y_pred[:, a : a * 2], \
                            y_pred[:, a * 2 : a * 3], y_pred[:, a * 3 : a * 4]
        return hinge_1(l_i, f_i)
        #return (hinge(l_i, f_i, f_k) + hinge(f_i, l_i, l_k)) / 2
        #return Cos_is(l_i, f_i) - Cos_is(l_i, f_k) - Cos_is(l_k, f_i)

    def Norm(X):
        return K.transpose(K.transpose(X) / (K.sqrt(tf.reduce_sum(K.square(X), 1) + 1e-9)))

    def shit_ik(X):
        l_i, l_k, f_i, f_k = Norm(X[:, a * 0 : a]), Norm(X[:, a : a * 2]), \
                            Norm(X[:, a * 2 : a * 3]), Norm(X[:, a * 3 : a * 4])
        return tf.concat([l_i, l_k, f_i, f_k], 1)

    print(" Begin ! ")

    ### Input
    input_feature = Input(shape = (word_num, num_feature, ), dtype = 'float32', name = 'input_feature')
    input_label = Input(shape = (1, ), dtype = 'int32', name = 'input_label')
    k_feature = Input(shape = (word_num, num_feature, ), dtype = 'float32', name = 'k_feature')
    k_label = Input(shape = (1, ), dtype = 'int32', name = 'k_label')

    ### Embed
    Embed = Embedding(input_dim = num_label, output_dim = dim_vector, input_length = 1, embeddings_initializer= 'glorot_normal')
    # embeddings_constraint = max_norm(max_value=2, axis=0))
    vector_label_i_1 = Embed(input_label)
    vector_label_k_1 = Embed(k_label)

    vector_label_i = Reshape((dim_vector, ))(vector_label_i_1)
    vector_label_k = Reshape((dim_vector, ))(vector_label_k_1)

    Dense_label_1 = Dense(dim_vector,activation='relu', kernel_initializer = 'glorot_normal')
    vector_label_i_1 = Dense_label_1(vector_label_i)
    vector_label_k_1 = Dense_label_1(vector_label_k)

    Dropout_2 = Dropout(0.2)
    vector_label_i_1_drop = Dropout_2(vector_label_i_1)
    vector_label_k_1_drop = Dropout_2(vector_label_k_1)

    Dense_label_2 = Dense(dim_vector, kernel_initializer = 'glorot_normal')
    vector_label_i_2 = Dense_label_2(vector_label_i_1_drop)
    vector_label_k_2 = Dense_label_2(vector_label_k_1_drop)

    ### Dense
    Dense_feature_1 = Dense(dim_vector,activation='relu')#, kernel_constraint = max_norm(max_value=2, axis=0))
    vector_feature_i_1 = Dense_feature_1(input_feature)
    vector_feature_k_1 = Dense_feature_1(k_feature)
    Dropout_1 = Dropout(0.2);
    vector_feature_i_1_drop = Dropout_1(vector_feature_i_1);
    vector_feature_k_1_drop = Dropout_1(vector_feature_k_1);

    LSTM_0 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=False)
    LSTM_1 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=True)
    LSTM_2 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=True)
    LSTM_3 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=True)
    LSTM_4 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=True)
    LSTM_5 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=True)
    LSTM_6 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=False)
    vector_feature_lstm_k = LSTM_0(k_feature);

    vector_feature_lstm_1 = LSTM_1(input_feature);
    vector_feature_1=vector_feature_lstm_1


    vector_feature_lstm_2=LSTM_2(vector_feature_1)
    vector_feature_2=Add()([vector_feature_1,vector_feature_lstm_2])

    vector_feature_lstm_3=LSTM_3(vector_feature_2)
    vector_feature_3=Add()([vector_feature_2,vector_feature_lstm_3])

    vector_feature_lstm_4=LSTM_4(vector_feature_3)
    vector_feature_4=Add()([vector_feature_3,vector_feature_lstm_4])

    vector_feature_lstm_5=LSTM_5(vector_feature_4)
    #vector_feature_lstm_i=Add([vector_feature_lstm_1,vector_feature_lstm_2,vector_feature_lstm_3,vector_feature_lstm_4,vector_feature_lstm_5])

    concat_feature=Concatenate(axis=1)([vector_feature_lstm_1,vector_feature_lstm_2,vector_feature_lstm_3,vector_feature_lstm_4,vector_feature_lstm_5])
    vector_feature_lstm_i=LSTM_6(concat_feature)

    Dense_feature_1 = Dense(dim_vector,activation='relu', kernel_initializer = 'glorot_normal')
    vector_feature_i_1 = Dense_feature_1(vector_feature_lstm_i)
    vector_feature_k_1 = Dense_feature_1(vector_feature_lstm_k)

    Dropout_1 = Dropout(0.2);
    vector_feature_i_1_drop = Dropout_1(vector_feature_i_1);
    vector_feature_k_1_drop = Dropout_1(vector_feature_k_1);

    Dense_feature_2 = Dense(dim_vector, kernel_initializer = 'glorot_normal')
    vector_feature_i = Dense_feature_2(vector_feature_i_1_drop)
    vector_feature_k = Dense_feature_2(vector_feature_k_1_drop)


    ### Loss1
    concat_ik = Concatenate(axis = 1)([vector_label_i_2, vector_label_k_2, vector_feature_i, vector_feature_k])
    IK = Lambda(shit_ik, output_shape = (dim_vector * 4, ), name = 'out_1')(concat_ik)

    ### Model
    self.model = Model(inputs = [input_feature, input_label, k_feature, k_label], outputs = [IK])

    ### Compile
    self.model.compile(loss = {'out_1' : Loss1}, loss_weights={'out_1' : 1000.}, optimizer = 'adam')

    #Save
    self.model.save_weights('/data/tmpsrt1/DCASE2017-baseline-system/applications/log_new/model_trivial_0.h5')
    #self.model.save('/data/tmpsrt1/DCASE2017-baseline-system/applications/log_new/jb.h5')

    '''

    num_feature = 100
    num_label = 15
    dim_vector = 128
    margin = 0.5
    batch_size = 32
    word_num = 1764
    dense_size = 128
    input_size = 501
    raw_size = 441001
    num_asd = 118
    wave_size = 32

    def my_loss(y_true, y_pred):
        '''
        a = y_pred[:,0:15]
        b = y_pred[:,15:-1]
        b = K.reshape(b,(-1, 15))
        print("rrrrrrrrrrrrrrrrrrrrrrrr")
        print("asavad ",a.get_shape)
        #a = K.repeat_elements(a,28*9,axis=0)
        '''

        print("asavad ",y_true.get_shape)
        print("asavad ",y_pred.get_shape)

        ans = []
        for i in range(num_asd):
            ans.append(K.categorical_crossentropy(y_true[:, i], y_pred[:,i]))
        return K.mean(tf.stack(ans))

    def mode(y_true, y_pred):
        a = K.argmax(y_pred, axis = 2)
        aa = K.argmax(y_true, axis = 2)
        ans = []
        for i in range(15):
            tmp = K.equal(a, i)
            #print("asavad ", tmp.get_shape)
            b = tf.reduce_sum(tf.cast(tmp, tf.int32), axis = 1, keep_dims=False)
            #print("asavad ", b.get_shape)
            ans.append(b)
        c = tf.stack(ans)
        d = K.argmax(K.transpose(c), axis = 1)
        return K.mean(K.equal(aa[:, 0], d))

    def mean_acc(y_true, y_pred):
        a = K.argmax(y_pred, axis = 2)
        aa = K.argmax(y_true, axis = 2)

        tmp = K.equal(a, aa)
        return K.mean(K.mean(tf.cast(tmp, tf.float32)))

    def func(X):
        return tf.reduce_sum(X,1)

    '''
    def shit(X):
        a = X[:,0:15]
        b = X[:,15:-1]
        b = K.reshape(b,(-1, 15))
        return K.categorical_crossentropy(a, b)
    '''

    '''
    ### Input
    input_feature = Input(shape = (input_size, num_feature, ), dtype = 'float32', name = 'input_feature')
    #input_feature = Input(shape = (num_feature, ), dtype = 'float32', name = 'input_feature')
    raw_feature = Input(shape = (raw_size, ), dtype = 'float32', name = 'raw_feature')
    #raw_feature = Input(shape = (num_feature, ), dtype = 'float32', name = 'raw_feature')
    '''



    '''
    specgram = Melspectrogram(n_dft=512,
                             input_shape=(raw_size, 2), 
                             trainable=True,
                             sr=11025)
    
    raw_spec = specgram(raw_feature)
    '''

    '''

    Conv_00 = Conv1D(128, 3, activation='relu', kernel_initializer = 'glorot_normal')
    Conv_01 = Conv1D(128, 3, activation='relu', kernel_initializer = 'glorot_normal')
    Pool_0 = MaxPooling1D(pool_size=2)
    conv_00 = Conv_00(input_feature)
    conv_01 = Conv_01(conv_00)
    pool_0 = Pool_0(conv_01)
    drop_0 = Dropout(0.2)(pool_0)

    Conv_10 = Conv1D(256, 3, activation='relu', kernel_initializer = 'glorot_normal')
    Conv_11 = Conv1D(256, 3, activation='relu', kernel_initializer = 'glorot_normal')
    Pool_1 = MaxPooling1D(pool_size=2)
    conv_10 = Conv_10(drop_0)
    conv_11 = Conv_11(conv_10)
    pool_1 = Pool_1(conv_11)
    drop_1 = Dropout(0.2)(pool_1)

    Conv_20 = Conv1D(512, 3, activation='relu', kernel_initializer = 'glorot_normal')
    Conv_21 = Conv1D(15, 3, activation='softmax', kernel_initializer = 'glorot_normal',name='out_1')
    conv_20 = Conv_20(drop_1)
    drop_2 = Dropout(0.25)(conv_20)
    vector_feature_i = Conv_21(drop_2)

    #Conv_02 = Conv1D(256, 3, padding='causal', activation='relu',dilation_rate=1, kernel_initializer = 'glorot_normal')
    '''

    '''
    ###!@3
    raw_feat=Reshape((raw_size,1,))(raw_feature)
    #print('vqe', raw_spec.get_shape())

    Conv_00 = Conv1D(256, 3, padding='causal', activation='relu',dilation_rate=1, kernel_initializer = 'glorot_normal')
    Conv_01 = Conv1D(256, 3, padding='causal', activation='relu',dilation_rate=1, kernel_initializer = 'glorot_normal')
    Conv_02 = Conv1D(256, 3, padding='causal', activation='relu',dilation_rate=1, kernel_initializer = 'glorot_normal')

    conv_00 = Conv_00(input_feature)
    conv_01 = Conv_01(conv_00)
    conv_02 = Conv_02(conv_01)
    batch_input = BatchNormalization()(conv_02)
    drop_0 = Dropout(0.2)(batch_input)

    Conv_03 = Conv1D(128, 3, activation='relu', kernel_initializer = 'glorot_normal')
    Conv_04 = Conv1D(15, 3, activation='softmax', kernel_initializer = 'glorot_normal')

    conv_03_input = Conv_03(drop_0)
    drop_03_input = Dropout(0.25)(conv_03_input)
    vector_input = Conv_04(drop_03_input)

    
    Conv_6 = Conv1D(8, 11, strides=11, kernel_initializer = 'glorot_normal')
    Conv_11 = Conv1D(16, 11,strides=10, kernel_initializer = 'glorot_normal')
    Conv_7 = Conv1D(wave_size, 11, strides=8, kernel_initializer = 'glorot_normal')
    #Conv_8 = Conv1D(32, 7, strides=3,kernel_initializer = 'glorot_normal')
    #Conv_12 = Conv1D(wave_size, 7, strides=3,kernel_initializer = 'glorot_normal')

    Conv_1 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=1, kernel_initializer = 'glorot_normal')
    Conv_2 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=2, kernel_initializer = 'glorot_normal')
    Conv_3 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=4, kernel_initializer = 'glorot_normal')
    Conv_4 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=8, kernel_initializer = 'glorot_normal')
    Conv_5 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=16, kernel_initializer = 'glorot_normal')
    Conv_13 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=32, kernel_initializer = 'glorot_normal')
    Conv_14 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=64, kernel_initializer = 'glorot_normal')
    Conv_15 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=128, kernel_initializer = 'glorot_normal')
    Conv_16 = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=256, kernel_initializer = 'glorot_normal')

    Conv_9 = Conv1D(128, 3, activation='relu')
    Conv_10 = Conv1D(15, 3, activation='softmax', name='out_1')

    conv_6 = Conv_6(raw_feat)
    conv_6_ok = LeakyReLU(alpha=.001)(conv_6)

    conv_11 = Conv_11(conv_6_ok)
    conv_11_ok = LeakyReLU(alpha=.001)(conv_11)

    conv_7 = Conv_7(conv_11_ok)
    conv_7_ok = LeakyReLU(alpha=.001)(conv_7)
    drop_7 = Dropout(0.1)(conv_7_ok)

    conv_1 = Conv_1(drop_7)
    conv_1_add = Add()([conv_1, drop_7])
    conv_2 = Conv_2(conv_1_add)
    conv_2_add = Add()([conv_2, conv_1_add])
    conv_3 = Conv_3(conv_2_add)
    conv_3_add = Add()([conv_3, conv_2_add])
    conv_4 = Conv_4(conv_3_add)
    conv_4_add = Add()([conv_4, conv_3_add])
    conv_5 = Conv_5(conv_4_add)
    conv_5_add = Add()([conv_5, conv_4_add])

    conv_13 = Conv_13(conv_5_add)
    conv_13_add = Add()([conv_13, conv_5_add])
    conv_14 = Conv_14(conv_13_add)
    conv_14_add = Add()([conv_14, conv_13_add])
    drop_14 = Dropout(0.2)(conv_14_add)

    Conv_9_input = Conv1D(128, 3, activation='relu', kernel_initializer = 'glorot_normal')
    Conv_10_input = Conv1D(15, 3, activation='softmax', kernel_initializer = 'glorot_normal')

    conv_9_input = Conv_9_input(drop_14)
    drop_9_input = Dropout(0.25)(conv_9_input)
    vector_raw = Conv_10_input(drop_9_input)
    
    vector_feature_i = Concatenate(axis = 1, name = 'out_1')([vector_raw, vector_input])

    '''

    '''
    #conv_1 = Conv_1(drop_12)
    #conv_1_add = Add()([conv_1, drop_12])
    #conv_2 = Conv_2(conv_1_add)
    #conv_2_add = Add()([conv_2, conv_1_add, drop_12])
    #conv_3 = Conv_3(conv_2_add)
    #conv_3_add = Add()([conv_3, conv_2_add, conv_1_add, drop_12])
    #conv_4 = Conv_4(conv_3_add)
    #conv_4_add = Add()([conv_4, conv_3_add, conv_2_add, conv_1_add, drop_12])
    #conv_5 = Conv_5(conv_4_add)
    #conv_5_add = Add()([conv_5, conv_4_add, conv_3_add, conv_2_add, conv_1_add, drop_12])
    #drop_5 = Dropout(0.2)(conv_5_add)
    '''

    '''

    
    '''
    '''
    Conv_1s = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=1, kernel_initializer = 'glorot_normal')
    Conv_2s = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=2, kernel_initializer = 'glorot_normal')
    Conv_3s = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=4, kernel_initializer = 'glorot_normal')
    Conv_4s = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=8, kernel_initializer = 'glorot_normal')
    Conv_5s = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=16, kernel_initializer = 'glorot_normal')
    Conv_13s = Conv1D(wave_size, 3, padding='causal', activation='relu',dilation_rate=32, kernel_initializer = 'glorot_normal')

    conv_1s = Conv_1s(conv_13)
    conv_1s_add = Add()([conv_1s, conv_13])
    conv_2s = Conv_2s(conv_1s_add)
    conv_2s_add = Add()([conv_2s, conv_1s_add])
    conv_3s = Conv_3s(conv_2s_add)
    conv_3s_add = Add()([conv_3s, conv_2s_add])
    conv_4s = Conv_4s(conv_3s_add)
    conv_4s_add = Add()([conv_4s, conv_3s_add])
    conv_5s = Conv_5s(conv_4s_add)
    #conv_5s_add = Add()([conv_5s, conv_4s_add])
    '''
    '''
    
    conv_13_add = Add()([conv_13,conv_5_add])
    conv_14 = Conv_14(conv_13_add)
    '''
    '''
    conv_14_add = Add()([conv_14,conv_13_add])
    conv_15 = Conv_15(conv_14_add)
    conv_15_add = Add()([conv_15,conv_14_add])
    conv_16 = Conv_16(conv_15_add)
    conv_16_add = Add()([conv_16,conv_15_add])
    '''
    '''
    drop_5 = Dropout(0.2)(conv_14)

    #res_1 = Add()([drop_12, drop_1])

    conv_9 = Conv_9(drop_5)
    drop_9 = Dropout(0.25)(conv_9)
    vector_feature_i = Conv_10(drop_9)
    '''



    '''
    Conv_1 = Conv2D(32, (3, 3), padding='same', activation='relu')
    Conv_2 = Conv2D(32, (3, 3), padding='same', activation='relu')
    Pool_1 = MaxPooling2D(pool_size=(1, 2))
    conv_1_input = Conv_1(input_feat)
    conv_2_input = Conv_2(conv_1_input)
    pool_1_input = Pool_1(conv_2_input)
    #drop_1_input = Dropout(0.2)(pool_1_input)

    Conv_3 = Conv2D(64, (3, 3), padding='same', activation='relu')
    Conv_4 = Conv2D(64, (3, 3), padding='same', activation='relu')
    Pool_2 = MaxPooling2D(pool_size=(2, 3))
    conv_3_input = Conv_3(pool_1_input)
    conv_4_input = Conv_4(conv_3_input)
    pool_2_input = Pool_2(conv_4_input)
    drop_2_input = Dropout(0.1)(pool_2_input)

    Conv_5 = Conv2D(64, (3, 3), padding='same', activation='relu')
    Conv_6 = Conv2D(64, (3, 3), padding='same', activation='relu')
    Pool_5 = MaxPooling2D(pool_size=(2, 3))
    Conv_11 = Conv2D(128, (3, 3), padding='same', activation='relu')
    Conv_7 = Conv2D(128, (3, 3), padding='same', activation='relu')
    Pool_3 = MaxPooling2D(pool_size=(1, 3))
    conv_5_input = Conv_5(drop_2_input)
    conv_6_input = Conv_6(conv_5_input)
    pool_5_input = Pool_5(conv_6_input)
    drop_5_input = Dropout(0.2)(pool_5_input)

    #pool_3_input = Pool_3(conv_6_input)
    #drop_3_input = Dropout(0.2)(pool_3_input)

    conv_11_input = Conv_11(drop_5_input)
    conv_7_input = Conv_7(conv_11_input)
    pool_3_input = Pool_3(conv_7_input)
    drop_4_input = Dropout(0.2)(pool_3_input)

    Conv_8 = Conv2D(256, (1, 1), padding='same', activation='relu')
    Conv_12 = Conv2D(256, (1, 1), padding='same', activation='relu')
    conv_8_input = Conv_8(drop_4_input)
    conv_12_input = Conv_12(conv_8_input)

    Pool_4 = AveragePooling2D(pool_size=(2, 2))
    pool_4_input = Pool_4(conv_12_input)
    drop_5_input = Dropout(0.2)(pool_4_input)

    #Flatten_input = Flatten()(pool_4_input)
    Conv_9 = Conv2D(256, (1, 1), padding='same', activation='relu')
    Conv_10 = Conv2D(15, (1, 1), padding='same', activation='softmax')
    conv_9_input = Conv_9(drop_5_input)
    drop_6_input = Dropout(0.2)(conv_9_input)
    print("fffffffffffffffffffffff")
    conv_10_input =Conv_10(drop_6_input)
    vector_feature_i = Reshape((-1, 15), name = 'out_1')(conv_10_input)
    #concat_1 = Concatenate(axis=1,name='out_1')([y_true,conv_10_input_re])
    #vector_feature_i = Lambda(shit,output_shape=(1,),name='out_1')(concat_1)
    '''

    #print("ffffffffffffffffffffffff")

    #Conv_9 = Conv2D(1, (1, 1),activation='relu')
    #conv_9_input = Conv_8(pool_4_input)
    #pool_4_input_1 = Reshape((28,9*64, ))(pool_4_input)
    #Conv_9 = Conv1D(15,1,activation='softmax')
    #conv_9_input = Conv_9(pool_4_input_1)
    #Conv_9 = Conv2D(15, (1, 1))
    #conv_9_input = Conv_9(pool_4_input)
    '''
    S1 = Permute((1,2))(conv_9_input)
    Soft = Activation('softmax')
    Soft_1_input = Soft(S1)
    S2 = Permute((1,2))(Soft_1_input)
    '''
    #Lam = Lambda(func, output_shape = (15, ))(conv_9_input)
    #Dense_1 = Dense(15, activation='softmax', name = 'out_1')
    #vector_feature_i = Dense_1(Lam)


    '''
    Conv_2 = Conv1D(dim_vector, 1, activation='relu')
    conv_2_input = Conv_2(conv_1_input)
    Dense(15, activation='softmax')

    fla = Flatten()
    fla_raw = fla(conv_2_input)
    
    
    Dense_2 = Dense(dim_vector,activation='relu')
    asd = Dense_2(input_feature)

    Dense_3 = Dense(dim_vector,activation='relu')
    cd = Dense_3(asd)
    


    Dense_4 = Dense(num_label, activation='softmax', name = 'out_1')
    vector_feature_i = Dense_4(cd)
    '''

    ### Model
    #self.model = Model(inputs = [raw_feature, input_feature], outputs = [vector_feature_i])

    ### Compile
    #self.model.compile(loss = {'out_1' : my_loss}, optimizer = 'adam', metrics=[mode, mean_acc])



    '''

    
    ### Dense
    Dense_1 = Dense(dim_vector,activation='relu', kernel_initializer = 'glorot_normal')
    feature_1 = Dense_1(input_feature)
    featuer_drop_1 = Dropout(0.2)(feature_1)

    ### LSTM
    LSTM_2 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=True)
    
    feature_tmp_1 = Reshape((1, dim_vector))(featuer_drop_1)
    #vector_feature_lstm_1 = LSTM_2(feature_tmp_1)
    
    concat_1 = Concatenate(axis = 1)([feature_tmp_1, vector_feature_lstm_1])
    
    vector_feature_lstm_2 = LSTM_2(concat_1)

    ### Dense
    Dense_2 = Dense(dim_vector,activation='relu', kernel_initializer = 'glorot_normal')
    feature_2 = Dense_2(featuer_drop_1)
    featuer_drop_2 = Dropout(0.2)(feature_2)

    ### LSTM
    LSTM_3 = LSTM(units = dim_vector, dropout = 0.2, activation='tanh', recurrent_activation='hard_sigmoid', \
                kernel_initializer='glorot_normal',return_sequences=False)

    feature_tmp_2 = Reshape((1, dim_vector))(featuer_drop_2)
    concat_2 = Concatenate(axis = 1)([feature_tmp_2, vector_feature_lstm_2])
    vector_feature_lstm_3 = LSTM_3(concat_2)
    
    ### Answer Dense
    Dense_3 = Dense(dense_size, activation='relu', kernel_initializer = 'glorot_normal')
    answer_3 = Dense_3(vector_feature_lstm_3)
    #answer_3=Dense_3(input_feature)
    answer_drop_3 = Dropout(0.2)(answer_3)

    Dense_4 = Dense(num_label, activation='softmax', kernel_initializer = 'glorot_normal', name = 'out_1')
    vector_feature_i = Dense_4(answer_drop_3)
    
    #Dense_4 = Dense(num_label, activation='softmax', kernel_initializer = 'glorot_normal', name = 'out_1')
    #vector_feature_i = Dense_4(vector_feature_lstm_1)
    ### Model
    self.model = Model(inputs = [raw_feature ], outputs = [vector_feature_i])

    ### Compile
    self.model.compile(loss = {'out_1' : 'categorical_crossentropy'}, optimizer = 'adam', metrics=["accuracy"])
    '''
    

    '''
    
    self.model = Sequential()
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(input_shape)
    #X=input_shape[0]
    #Y=input_shape[1]
    self.model.add(LSTM(256,input_shape=(10,200)))
    self.model.add(Dense(512,activation='relu'))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(15,activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])

    '''

    '''
    self.model.add(Conv1D(256, 3, activation='relu', input_shape=(501,200)))
    self.model.add(Dropout(0.25))
    #self.model.add(BatchNormalization())
    self.model.add(Conv1D(512, 3, activation='relu'))
    #self.model.add(BatchNormalization())
    #self.model.add(MaxPooling1D(pool_size=2))
    self.model.add(Dropout(0.25))
    self.model.add(Flatten())
    self.model.add(Dense(15, activation='softmax'))
    #self.model.add(Conv1D(64, 3, activation='relu'))
    #self.model.add(BatchNormalization())
    #self.model.add(Conv1D(64, 3, activation='relu'))
    #self.model.add(BatchNormalization())
    #self.model.add(MaxPooling1D(pool_size=2))
    #self.model.add(Dropout(0.25))
    

    self.model.add(Dense(64,activation='relu',input_shape=(501,200)))
    self.model.add(Dropout(0.25))
    #self.model.add(Dense(64,activation='relu',input_shape=(501,200)))
    self.model.add(Flatten())
    self.model.add(Dense(256, activation='relu'))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(15, activation='softmax'))
    
    
    self.model.add(Conv1D(256, 3, activation='relu', input_shape=(501,200)))
    self.model.add(LSTM(512))
    
    self.model.add(Dense(1024,activation='relu'))
    self.model.add(Dropout(0.25))        
    self.model.add(Dense(15, activation='softmax'))
    
    self.model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    '''

    

