import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import coremltools as ct
from coremltools.proto import FeatureTypes_pb2 as ft
import argparse



def parse_args():
    desc = "AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i','--pb_model_path', type=str, help='The path of the input pb file')
    parser.add_argument('-o','--out_ml', type=str, help='The path of the output mlmodel file')
    parser.add_argument('-iw','--input_w', type=int, default=512, help='The width of input image')
    parser.add_argument('-ih','--input_h', type=int, default=512, help='The height of input image')
    return parser.parse_args()


"""
convert pb model (from tf-1.15.0) into CoreML .mlmodel (apple use)
"""

def toml(pbfile, modelname, input_w, input_h): # in AnimeGANv3 the name is : AnimeGANv3_input or animeganv3_input. the name key-value can be omitted.
    """

    :param pbfile: tf generated pb model file (str)
    :param modelname: The output coreml model file (str)
    :param input_w: input image width (int)
    :param input_h: input image height (int)
    """
    mlmodel = ct.convert(model=pbfile, convert_to="neuralnetwork", source="tensorflow", inputs=[ct.ImageType( channel_first=False, shape=[1,input_h,input_w,3],bias=[-1,-1,-1], scale=2.0/255.0)],)
    # mlmodel = ct.convert(model=ptfile, convert_to="neuralnetwork", inputs=[ct.ImageType( channel_first=True, shape=[1, 3, input_h,input_w],bias=[-1,-1,-1], scale=2.0/255.0)]) # For pytorch, the .pt model, channel_first=True
    
    ## Set description, author, version number
    mlmodel.description = "AnimeGANv3_model"
    mlmodel.author = "Asher Chan"
    mlmodel.version = "v1.0.0"
    mlmodel.save("temp.mlmodel") # Intermediate Temporary Model
    spec = ct.utils.load_spec("temp.mlmodel")
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)

    print(builder.spec.description)
    builder.add_permute(name="permute", dim=[0, 3, 1, 2], input_name= "generator_1_main_out_layer", # "generator_main_out_layer" ( output name of v3 face style)
                        output_name="permute_out")
    builder.add_squeeze(name="squeeze", input_name="permute_out", output_name="squeeze_out", axes=None,
                        squeeze_all=True)
    builder.add_activation(name="activation", non_linearity="LINEAR", input_name="squeeze_out", output_name="image",params=[127.5, 127.5])
    builder.spec.description.output.pop()
    builder.spec.description.output.add()
    output = builder.spec.description.output[0]
    output.name = "image"
    output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('RGB')
    output.type.imageType.width = input_w
    output.type.imageType.height = input_h
    ct.utils.save_spec(builder.spec, modelname)


def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()
    toml(args.pb_model_path , args.out_ml ,args.input_w , args.input_h)
    
    # print info
    spec = ct.utils.load_spec(args.out_ml)
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
    print(builder.spec.description)


    """
     Model prediction with .mlmodel is only supported on macOS version 10.13 or later.
    """
    # import PIL
    # import numpy as np
    # def load_image(path, resize_to=None):
    #     # resize_to: (Width, Height)
    #     img = PIL.Image.open(path)
    #     if resize_to is not None:
    #         img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    #     img_np = np.array(img).astype(np.float32)
    #     return img_np, img
    #
    # model = ct.models.MLModel(args.out_ml)
    # img_np, img = load_image('img.png', resize_to=(args.input_w , args.input_h))
    # out_dict = model.predict({'AnimeGANv3_input': img_np})


if __name__ == '__main__':
    main()
  