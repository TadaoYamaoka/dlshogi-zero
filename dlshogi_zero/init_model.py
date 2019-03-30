from dlshogi_zero.nn.resnet import ResNet

def init_model(model_path):
    model = ResNet()
    model.save(model_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', default='model-000.h5')
    args = parser.parse_args()

    init_model(args.model_path)
