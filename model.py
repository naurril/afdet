import tensorflow as tf
import cfg


def conv2d(x, filters, kernel_size, strides, is_training=True):
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x, is_training)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def point_pillars(pillars, is_training):
    # split
    # B * H * W  * N * D
    # B * P * N * D
    x = tf.reshape(pillars, [-1, pillars.shape[1]*pillars.shape[2], pillars.shape[3], pillars.shape[4]])
    x = conv2d(x, 64, (1,1), (1,1), is_training)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, x.shape[2]))(x)   # b pillar 1 d=64
    x = tf.reshape(x, [-1, pillars.shape[1], pillars.shape[2], x.shape[3]])   # b h w d3=64, each pillar is represented by a 64-vector
    return x

def backbone(x, is_training):
    # 64 channels
    for _ in range(7):
        x = conv2d(x, 32, (3,3), (1,1), is_training)

    phase1 = conv2d(x, 64, (1,1), (1,1), is_training)

    x = conv2d(x, 64, (3,3), (2,2), is_training)
    for _ in range(7):
        x = conv2d(x, 32, (3,3), (1,1), is_training)
    #
    x = tf.keras.layers.Conv2DTranspose(64, (2,2),(2,2))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x, is_training)
    x = tf.keras.layers.ReLU()(x)

    x = tf.concat([phase1, x], axis=-1, name="bakcbone_end")
    return x


def get_header(input, num_classes, is_training):
    # cannot find description of the hotmap header
    # b, h, w, d
    x = input
    #x = conv2d(x, num_classes, (3,3),(1,1), is_training)
    #x = conv2d(x, num_classes, (1,1),(1,1), is_training)
    x = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(num_classes, (1,1), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.clip_by_value(x,-5,5)

    x = tf.math.sigmoid(x, name="heatmap")
    heatmap = x

    x = input
    #x = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(2, (1,1), strides=(1,1), padding='same', data_format='channels_last')(x)
    offset = x

    # x = input
    # x = conv2d(x, 64, (1,1),(1,1), is_training)
    # x = conv2d(x, 32, (1,1),(1,1), is_training)
    # x = conv2d(x, num_classes, (1,1),(1,1), is_training)    
    # z_value = x

    # x = input
    # x = conv2d(x, 64, (1,1),(1,1), is_training)
    # x = conv2d(x, 32, (1,1),(1,1), is_training)
    # x = conv2d(x, num_classes*3, (1,1),(1,1), is_training)
    # dim = x

    # x = input
    # x = conv2d(x, 64, (1,1),(1,1), is_training)
    # x = conv2d(x, 32, (1,1),(1,1), is_training)
    # x = conv2d(x, num_classes*8, (1,1),(1,1), is_training)
    # orientation = x
    
    return tf.concat([heatmap, offset], axis=-1)



## eq (2) in afdet paper
alpha = 2
beta = 4

def heatmap_loss(y_gt, y_pred, y_ind):
    # b, h, w, num_cls
    cond = (y_ind == 1.0)
    norm = tf.reduce_sum(y_ind, axis=[1,2,3]) # count the number object centers.

    #y_pred = tf.math.sigmoid(y_pred)
    ele_loss = tf.where(cond,
        tf.math.pow(1-y_pred, alpha) * tf.math.log(y_pred),
        tf.math.pow(1-y_gt, beta) * tf.math.pow(y_pred, alpha) * tf.math.log(-y_pred+1))
    
    sum = tf.reduce_mean(ele_loss, axis=[1,2,3])
    
    loss = -sum/(norm+1)
    #loss = -sum
    loss = tf.reduce_mean(loss)
    return loss
def offset_loss(y_gt, y_pred):

    y_ind = tf.where(y_gt!=0.0, 1.0, 0.0)
    return index_regressoin_l1_loss(y_gt, y_pred, y_ind)

def index_regressoin_l1_loss(y_gt, y_pred, y_ind):
    # backprop only positions where object exists, 
    # so we need to input a index parameter, can we do it in tf.keras?
    # we use gt!= 0.0 as the index, note this is not precise.
    cond = (y_ind == 1.0)
    norm = 1 + tf.reduce_sum(y_ind, axis=[1,2,3],keepdims=True) # count the number object centers.
    
    ele_loss = tf.where(cond, tf.abs(y_gt - y_pred)/norm, 0)
    
    loss = tf.reduce_sum(ele_loss, axis=[1,2,3])
    
    loss = tf.reduce_mean(loss)
    return loss

def orientation_regression_loss(y_gt, y_pred, y_ind):
    # g_pred:
    # a,b,c,d,  a,b,c,d
    # a,b softmax
    # c,d cos/sin

    # b,h,w,2

    # y_gt: in_bin1, in_bin2, radian 
    # b,h,w,3

    in_bin1 = y_gt[:,:,:,0]
    in_bin2 = y_gt[:,:,:,1]    
    radian  = y_gt[:,:,:,2]

    bin1_logit = y_pred[:,:,:,:2]
    bin1_logit_gt = tf.stack([in_bin1, 1-in_bin1],axis=-1)
    bin1_crossentropy = tf.losses.binary_crossentropy(bin1_logit_gt,  bin1_logit, from_logits=True)
    bin1_cos_sin_pred   = y_pred[:,:,:,2:4]
    bin1_cos_sin_gt = tf.stack([tf.math.cos(radian), tf.math.sin(radian)], axis=-1)
    bin1_radian_loss = tf.losses.mean_squared_error(bin1_cos_sin_gt, bin1_cos_sin_pred)

    bin2_logit = y_pred[:,:,:,4:6]
    bin2_logit_gt = tf.stack([in_bin2, 1-in_bin2],axis=-1)
    bin2_crossentropy = tf.losses.categorical_crossentropy(bin2_logit_gt,  bin2_logit, from_logits=True)

    bin2_cos_sin_pred   = y_pred[:,:,:,6:8]
    bin2_cos_sin_gt = bin1_cos_sin_gt    
    bin2_radian_loss = tf.losses.mean_squared_error(bin2_cos_sin_gt, bin2_cos_sin_pred)
    
    
    loss = tf.stack([bin1_crossentropy, bin1_radian_loss, bin2_crossentropy, bin2_radian_loss], axis=-1)

    loss = tf.where(y_ind, loss, 0)
    loss = tf.reduce_mean(loss)
    return loss



class TfnetLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        input_obj_ind = y_true[:,:,:,0:1]
        input_heatmap = y_true[:,:,:,1:(cfg.CLASS_NUM+1)]
        input_offset  = y_true[:,:,:,(cfg.CLASS_NUM+1):(cfg.CLASS_NUM+1+2)]

        heatmap = y_pred[:,:,:,0:cfg.CLASS_NUM]
        offset  = y_pred[:,:,:,cfg.CLASS_NUM:(cfg.CLASS_NUM+2)]

        #tf.summary.image("heatmap", tf.stack([input_heatmap,heatmap], axis=-1), step=0)

        tf.summary.image("pred-heatmap", heatmap, step = 0)
        tf.summary.image("gt-heatmap", input_heatmap, step = 0)

        loss1 = heatmap_loss(input_heatmap, heatmap, input_obj_ind)
        loss2 = offset_loss(input_offset, offset)
        #loss = tf.reduce_mean(tf.square(input_heatmap-heatmap))
        
        return loss2 + loss1


def get_model(num_classes, input_dim, is_training):
    H,W,P,D=input_dim
    
    input_pointcloud = tf.keras.Input(shape=(H, W, P, D-3)) #
    
    # input_obj_ind = tf.keras.Input(dtype=tf.bool, shape=(H, W, 1)) #

    # #ground truth
    # input_heatmap = tf.keras.Input(shape=(H, W, num_classes))
    
    # input_offset =  tf.keras.Input(shape=(H, W, 2))
    
    # input_z_value =  tf.keras.Input(shape=(H, W, num_classes*1))
    # input_dim =  tf.keras.Input(shape=(H, W, num_classes*3))
    # input_orientation = tf.keras.Input(shape=(H, W, num_classes*8))

    x = point_pillars(input_pointcloud, is_training)
    x = backbone(x, is_training)
    header = get_header(x, num_classes, is_training)
    #(heatmap, offset, z_value, dim, orientation) = header
    #(heatmap, offset) = header
    #output = tf.concat(header, axis=-1, name="output")
    output = header
    # so loss is another graph part
    # loss = heatmap_loss(input_heatmap, heatmap) + \
    #        index_regressoin_l1_loss(input_offset, offset, input_obj_ind)# +  \
    #     #    index_regressoin_l1_loss(input_z_value, z_value, input_obj_ind) +  \
    #     #    index_regressoin_l1_loss(input_dim, dim, input_obj_ind) + \
    #     #    orientation_regression_loss(input_orientation, orientation, input_obj_ind) + \
           
    model = tf.keras.Model(inputs=[input_pointcloud], #input_z_value, input_dim, input_orientation], 
                           outputs=[output])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), metrics=[], loss=TfnetLoss())#tf.keras.losses.mse)
    model.summary()

    return model
