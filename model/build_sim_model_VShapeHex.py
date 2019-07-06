import util

# Keras imports
from keras.layers import Input, Convolution1D, MaxPooling1D, Merge, Dropout, Flatten, Dense, BatchNormalization, LSTM, \
    Activation, Bidirectional, Reshape, AveragePooling1D
from keras.layers import concatenate
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import Sequential, Model
from keras.regularizers import l1, l2
from keras import backend as K
from keras.layers.core import Lambda

# model parameters
enhancer_length = 1600  # TODO: get this from input
promoter_length = 3000  # TODO: get this from input
n_kernels = 300  # Number of kernels; used to be 1024
filter_length = 40  # Length of each kernel
LSTM_out_dim = 50  # Output direction of ONE DIRECTION of LSTM; used to be 512
dense_layer_size = 800
mlp_layer_size = 20

# Convolutional/maxpooling layers to extract prominent motifs
# Separate identically initialized convolutional layers are trained for
# enhancers and promoters
# Define enhancer layers
enhancer_conv_layer = Convolution1D(input_dim=4,
                                    input_length=enhancer_length,
                                    nb_filter=n_kernels,
                                    filter_length=filter_length,
                                    border_mode="same",
                                    subsample_length=1,
                                    W_regularizer=l2(1e-5))
enhancer_max_pool_layer = MaxPooling1D(pool_length=int(filter_length / 2), stride=int(filter_length / 2), padding="valid")

'''
enhancer_length_slim = enhancer_length+filter_length-1
n_kernels_slim = 200
filter_length_slim = 20
enhancer_conv_layer_slim = Convolution1D(input_dim=4,
                                    input_length=enhancer_length_slim,
                                    nb_filter=n_kernels_slim,
                                    filter_length=filter_length_slim,
                                    border_mode="valid",
                                    subsample_length=1,
                                    W_regularizer=l2(1e-5))
'''
# Build enhancer branch
enhancer_branch = Sequential()
enhancer_branch.add(enhancer_conv_layer)
enhancer_branch.add(Activation("relu"))
#enhancer_branch.add(enhancer_conv_layer_slim)
#enhancer_branch.add(Activation("relu"))
enhancer_branch.add(enhancer_max_pool_layer)

# Define promoter layers branch:
promoter_conv_layer = Convolution1D(input_dim=4,
                                    input_length=promoter_length,
                                    nb_filter=n_kernels,
                                    filter_length=filter_length,
                                    border_mode="same",
                                    subsample_length=1,
                                    W_regularizer=l2(1e-5))
promoter_max_pool_layer = MaxPooling1D(pool_length=int(filter_length / 2), stride=int(filter_length / 2), padding="valid")

'''
promoter_length_slim = promoter_length+filter_length-1
n_kernels_slim = 200
filter_length_slim = 20
promoter_conv_layer_slim = Convolution1D(input_dim=4,
                                    input_length=promoter_length_slim,
                                    nb_filter=n_kernels_slim,
                                    filter_length=filter_length_slim,
                                    border_mode="valid",
                                    subsample_length=1,
                                    W_regularizer=l2(1e-5))
'''

# Build promoter branch
promoter_branch = Sequential()
promoter_branch.add(promoter_conv_layer)
promoter_branch.add(Activation("relu"))
#promoter_branch.add(promoter_conv_layer_slim)
#promoter_branch.add(Activation("relu"))
promoter_branch.add(promoter_max_pool_layer)

#fyt encoder
# Define main model layers
# Concatenate outputs of enhancer and promoter convolutional layers
merge_layer = Merge([enhancer_branch, promoter_branch],
                    mode='concat',
                    concat_axis=1)

'''
# Bidirectional LSTM to extract combinations of motifs
biLSTM_layer = Bidirectional(LSTM(input_dim=n_kernels,
                                  output_dim=LSTM_out_dim,
                                  return_sequences=True))
'''
# Dense layer to allow nonlinearities
dense_layer = Dense(output_dim=dense_layer_size,
                    init="glorot_uniform",
                    W_regularizer=l2(1e-6))

# Logistic regression layer to make final binary prediction
LR_classifier_layer = Dense(output_dim=1)


enhancer_mlp_layer = Dense( output_dim=mlp_layer_size,
                            init="glorot_uniform",
                            W_regularizer=l2(1e-6), 
                            )

promoter_mlp_layer = Dense( output_dim=mlp_layer_size,
                            init="glorot_uniform",
                            W_regularizer=l2(1e-6), 
                            )

mlp_pooling = AveragePooling1D(pool_length=int(filter_length / 2), stride=int(filter_length / 2), padding="valid")


def build_model(use_JASPAR=True):


    enhancer_input = Input((1600, 4))
    promoter_input = Input((3000, 4))



    enhancer_branch_encoder = enhancer_branch(enhancer_input)
    promoter_branch_encoder = promoter_branch(promoter_input)
    encoder = concatenate([enhancer_branch_encoder, promoter_branch_encoder], axis=1)
    encoder = BatchNormalization()(encoder)
    encoder = Dropout(0.25)(encoder)
    encoder = Flatten()(encoder)
    encoder = dense_layer(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation("relu")(encoder)
    encoder = Dropout(0.2)(encoder)
    #print(encoder.shape)
    #assert(1==2)
    #test
    #ans = LR_classifier_layer(encoder)
    #ans = BatchNormalization()(ans)
    #ans = Activation("sigmoid")(ans)
    #print(ans.shape)
    #assert(1==0)


    # A single downstream model merges the enhancer and promoter branches
    # Build main (merged) branch
    # Using batch normalization seems to inhibit retraining, probably because the
    # point of retraining is to learn (external) covariate shift
    #model = Sequential()
    #model.add(merge_layer)
    #model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    #model.add(biLSTM_layer)
    #model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    #model.add(Flatten())
    #model.add(dense_layer)
    #model.add(BatchNormalization())
    #model.add(Activation("relu"))
    #model.add(Dropout(0.2))
    #model.add(LR_classifier_layer)
    #model.add(BatchNormalization())
    #model.add(Activation("sigmoid"))
    
    enhancer_mlp = mlp_pooling(enhancer_input)
    promoter_mlp = mlp_pooling(promoter_input)
    #print(enhancer_mlp.shape)
    #print(promoter_mlp.shape)
    #assert(1==0)

    enhancer_mlp = Flatten()(enhancer_mlp)
    promoter_mlp = Flatten()(promoter_mlp)


    enhancer_mlp = enhancer_mlp_layer(enhancer_mlp)
    enhancer_mlp = BatchNormalization()(enhancer_mlp)
    enhancer_mlp = Activation("relu")(enhancer_mlp)
    enhancer_mlp = Dropout(0.2)(enhancer_mlp)
    

    promoter_mlp = promoter_mlp_layer(promoter_mlp)
    promoter_mlp = BatchNormalization()(promoter_mlp)
    promoter_mlp = Activation("relu")(promoter_mlp)
    promoter_mlp = Dropout(0.2)(promoter_mlp)
    
    mlp = concatenate([enhancer_mlp, promoter_mlp], axis=1)
    mlp = BatchNormalization()(mlp)



    yconv_contact_loss = concatenate([encoder, mlp], axis=1)
    pad = K.zeros_like(mlp, K.tf.float32)
    yconv_contact_pred = concatenate([encoder, pad], 1)
    pad2 = K.zeros_like(encoder, K.tf.float32)
    yconv_contact_H = concatenate([pad2, mlp], 1)

    y_conv_loss = LR_classifier_layer(yconv_contact_loss)
    y_conv_pred = LR_classifier_layer(yconv_contact_pred)
    y_conv_H = LR_classifier_layer(yconv_contact_H)


    #decoder = y_conv_loss - K.tf.matmul(K.tf.matmul(K.tf.matmul(y_conv_H, K.tf.linalg.inv(K.tf.matmul(y_conv_H, y_conv_H, transpose_a=True))),
                              #y_conv_H, transpose_b=True), y_conv_loss)

    #decoder = Lambda(lambda y_conv_loss, y_conv_H:
        #y_conv_loss - K.tf.matmul(K.tf.matmul(K.tf.matmul(y_conv_H, K.tf.linalg.inv(K.tf.matmul(y_conv_H, y_conv_H, transpose_a=True))),y_conv_H, transpose_b=True), y_conv_loss))(y_conv_loss,y_conv_H)

    decoder = Lambda(lambda x:
        K.tf.matmul(K.tf.matmul(x, K.tf.linalg.inv(K.tf.matmul(x, x, transpose_a=True))),x, transpose_b=True)
        )(y_conv_H)
    
    decoder = Lambda(lambda y:
        y - K.tf.matmul(decoder, y)
        )(y_conv_loss)
    
    #decoder = Reshape(())(decoder)
    #print(decoder.shape)
    #assert(1==0)
    # Read in and initialize convolutional layers with motifs from JASPAR
    if use_JASPAR:
        util.initialize_with_JASPAR(enhancer_conv_layer, promoter_conv_layer)
    
    decoder = Activation('sigmoid')(decoder)
    
    #answer = model([enhancer_input, promoter_input])
    model = Model([enhancer_input, promoter_input], decoder)
    #model = Model([enhancer_input, promoter_input], ans) #test

    return model


