IMAGE_SIZE = 256
BATCH_SIZE = 16

def load_data(image_path):
    print(image_path)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    return image

def data_generator(low_light_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

train_low_light_images = sorted(glob("/content/gdrive/MyDrive/Image_enhancement/Dataset/Train/AutoContrastLow_Mask_Ignore/*"))
val_low_light_images = sorted(glob("/content/gdrive/MyDrive/Image_enhancement/Dataset/Test/Low/*"))
test_low_light_images = sorted(glob("/content/gdrive/MyDrive/Image_enhancement/Dataset/Test/Low/*"))

train_dataset = data_generator(train_low_light_images)
val_dataset = data_generator(val_low_light_images)

print("Train Dataset:", train_dataset)
print("Validation Dataset:", val_dataset)
