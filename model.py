import tensorflow as tf
import config


def conv2d(x, filters, kernel_size, strides, is_training=True):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x, is_training)
    x = tf.keras.layers.ReLU()(x)
    return x

def point_pillars(pillars, is_training):
    # split
    # B * H * W  * N * D
    # B * P * N * D
    x = tf.reshape(pillars, [-1, pillars.shape[1]*pillars.shape[2], pillars.shape[3], pillars.shape[4]])
    #x = conv2d(x, 64, (1,1), (1,1), is_training)
    #x = pillars
    x = tf.keras.layers.Conv2D(64, 1, strides=1, padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x, is_training)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(1, x.shape[2]))(x)   # b pillar 1 d=64
    x = tf.reshape(x, [-1, pillars.shape[1], pillars.shape[2], x.shape[3]])   # b h w d3=64, each pillar is represented by a 64-vector
    #x = tf.squeeze(x, axis=-2)
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

def get_multiple_headers(input, num_classes, is_training):
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

    x = input
    #x = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(1, (1,1), strides=(1,1), padding='same', data_format='channels_last')(x)
    z = x

    x = input
    #x = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(3, (1,1), strides=(1,1), padding='same', data_format='channels_last')(x)
    size = x

    x = input
    #x = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(2, (1,1), strides=(1,1), padding='same', data_format='channels_last')(x)
    angle = x

    return heatmap, offset, z, size, angle

def get_header(input, num_classes, is_training):
   
    heatmap, offset, z, size, angle = get_multiple_headers(input, num_classes, is_training)
    return tf.concat([heatmap, offset, z, size, angle], axis=-1)



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
    
    sum = tf.reduce_sum(ele_loss, axis=[1,2,3])
    
    loss = -sum/(norm+1)
    #loss = -sum
    loss = tf.reduce_mean(loss)
    return loss
def offset_loss(y_gt, y_pred):
    y_ind = tf.where(y_gt!=0.0, 1.0, 0.0)
    return index_masked_regressoin_l1_loss(y_gt, y_pred, y_ind)

def index_masked_regressoin_l1_loss(y_gt, y_pred, y_ind):
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
        print(y_true.shape, y_pred.shape)
        input_obj_ind = y_true[:,:,:,0:1]
        input_heatmap = y_true[:,:,:,1:(config.CLASS_NUM+1)]
        input_offset  = y_true[:,:,:,(config.CLASS_NUM+1):(config.CLASS_NUM+3)]
        input_z = y_true[:,:,:,(config.CLASS_NUM+3):(config.CLASS_NUM+4)]
        input_size = y_true[:,:,:,(config.CLASS_NUM+4):(config.CLASS_NUM+7)]
        input_angle = y_true[:,:,:,(config.CLASS_NUM+7):(config.CLASS_NUM+9)]

        pred_heatmap = y_pred[:,:,:,0:config.CLASS_NUM]
        pred_offset  = y_pred[:,:,:,config.CLASS_NUM:(config.CLASS_NUM+2)]
        pred_z  = y_pred[:,:,:,(config.CLASS_NUM+2):(config.CLASS_NUM+3)]
        pred_size  = y_pred[:,:,:,(config.CLASS_NUM+3):(config.CLASS_NUM+6)]
        pred_angle  = y_pred[:,:,:,(config.CLASS_NUM+6):(config.CLASS_NUM+8)]

        tf.summary.image("pred-heatmap", pred_heatmap)
        tf.summary.image("gt-heatmap", input_heatmap)

        loss_heatmap = heatmap_loss(input_heatmap, pred_heatmap, input_obj_ind)
        loss_offset =  offset_loss(input_offset, pred_offset)
        loss_z = index_masked_regressoin_l1_loss(input_z, pred_z, input_obj_ind)
        loss_size = index_masked_regressoin_l1_loss(input_size, pred_size, input_obj_ind)
        loss_angle = index_masked_regressoin_l1_loss(input_angle, pred_angle, input_obj_ind)
        
        
        tf.summary.scalar("loss_heatmap", loss_heatmap)
        tf.summary.scalar("loss_offset", loss_offset)
        tf.summary.scalar("loss_z", loss_z)
        tf.summary.scalar("loss_size", loss_size)
        tf.summary.scalar("loss_angle", loss_angle)
        
        loss_heatmap = loss_heatmap*0.5
        return loss_heatmap + loss_offset + loss_z + loss_size + loss_angle


class AfdetModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        pillars, coord, gt = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

def get_model(num_classes, input_dim, is_training):
    N,H,W,P,D=input_dim
    
    #input_pillars = tf.keras.Input(shape=[N,P,D])
    #input_indices = tf.keras.Input(shape=[N,2], dtype=tf.int64)
    input_pointcloud = tf.keras.Input(shape=(H, W, P, D)) #
    
    # input_obj_ind = tf.keras.Input(dtype=tf.bool, shape=(H, W, 1)) #

    # #ground truth
    # input_heatmap = tf.keras.Input(shape=(H, W, num_classes))
    
    # input_offset =  tf.keras.Input(shape=(H, W, 2))
    
    # input_z_value =  tf.keras.Input(shape=(H, W, num_classes*1))
    # input_dim =  tf.keras.Input(shape=(H, W, num_classes*3))
    # input_orientation = tf.keras.Input(shape=(H, W, num_classes*8))

    x = point_pillars(input_pointcloud, is_training)   # x is 1600*64

    # feature_dim = x.shape[-1]
    # def scatter_pillars(arg):
    #     indices, pillars = arg
    #     return tf.scatter_nd(indices, pillars, [H,W,feature_dim])
    # x = tf.vectorized_map(scatter_pillars, (input_indices, x))

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
