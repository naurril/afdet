import tensorflow as tf
import model as M
import dataset
import datetime

model_file = "afnet.h5"
weights_file = "afnet_weights.h5"

from config import *


tf.random.set_seed(0)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()


model = M.get_model(1, [H, W, P, D], True)

train_data, eval_data = dataset.get_dataset()
train_data = train_data.batch(4)
eval_data = eval_data.batch(4)


tf.debugging.experimental.enable_dump_debug_info(
    log_dir,
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)


def lr_scheduler(epoch):
    def lr_by_epoch(epoch):
        if epoch < 10:
            return 0.001
        elif epoch < 5:
            return 0.0005
        elif epoch < 20:
            return 0.0001
        elif epoch < 40:
            return 0.00005
        else: 
            return 0.00001
    lr = lr_by_epoch(epoch)
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr
    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1)

class SaveCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            self.model.save("epoch_{}_{}".format(epoch, model_file), include_optimizer=True, overwrite=True)
            model.save_weights("epoch_{}_{}".format(epoch, weights_file))
            print("model saved!")


#model.fit(train_data, validation_data=train_data , epochs=250, callbacks=[tensorboard_callback, lr_callback, SaveCallback()])
model.fit(train_data, validation_data=eval_data, epochs=80, callbacks=[tensorboard_callback, lr_callback, SaveCallback()])

model.save(model_file, include_optimizer=True, overwrite=True)
model.save_weights(weights_file)
