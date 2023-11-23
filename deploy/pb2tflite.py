import tensorflow as tf
import os,cv2
import numpy as np

# tensorflow2.x
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pb2tflite(input_pb, input_size):


    converter = tf.lite.TFLiteConverter.from_frozen_graph(input_pb, #path to the pb file
                                                                  input_arrays=["AnimeGANv3_input"],  # The name of the model input node
                                                                  input_shapes={'AnimeGANv3_input': [1, input_size[0], input_size[1], 3]},
                                                                  output_arrays=["generator_1/main/out_layer"])  # The name of the model output node


    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(f"{input_pb.rsplit('.', 1)[0]}.tflite", "wb").write(tflite_model)  # save  tflite

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input = interpreter.get_input_details()
    print(input)
    output = interpreter.get_output_details()
    print(output)

    return f"{input_pb.rsplit('.', 1)[0]}.tflite"


class test:
    def __init__(self, tflite_path, input_size):
        self.input_size = input_size
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, img):
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        img = img.astype(np.float32) / 127.5 - 1.0
        return np.expand_dims(img, axis=0)

    def post_process(self, output):
        image = (np.squeeze(output) + 1.) / 2 * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def run(self, img):
        input_data = self.preprocess(img)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_data = self.post_process(output_data)
        return output_data



if __name__ == '__main__':
    input_size = [512, 512]  # height,width
    pb_path = r"AnimeGANv3_Hayao_36.pb"
    tflite_file = pb2tflite(pb_path, input_size)

    img = cv2.imread('img.png')[:,:,::-1] # bgr->rgb
    test_obj = test(tflite_file, input_size)
    out = test_obj.run(img)
    cv2.imwrite('res.jpg', out[:,:,::-1] )


