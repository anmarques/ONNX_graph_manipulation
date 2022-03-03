import argparse
import os
import onnxruntime as ort
import onnx
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(description='Evaluates onnx model')
    parser.add_argument('-i', '--input', type=str, help='input graph file name')
    parser.add_argument('-r', '--reference', type=str, default='', help='reference graph file name')
    parser.add_argument('-s', '--input_shape', type=int, action="extend", nargs="+", default=[1, 3, 224, 224],
                        help='input shape')
    parser.add_argument('-d', '--dataset', type=str, default='random', help='evaluation dataset')
    parser.add_argument('-p', '--dataset_path', type=str, default='', help='path to evaluation dataset')
    args = parser.parse_args()

    assert args.dataset in ['random', 'images']

    model = onnx.load(args.input)
    onnx.checker.check_model(model)

    if args.reference:
        reference_model = onnx.load(args.reference)
        onnx.checker.check_model(reference_model)

    if args.dataset == 'random':
        model_sess = ort.InferenceSession(args.input)
        x = (np.random.rand(*args.input_shape) * 255. - 127.5).astype(np.float32)
        outputs = model_sess.run(None, {'input': x})[0]

        if args.reference:
            ref_sess = ort.InferenceSession(args.reference)
            reference_outputs = ref_sess.run(None, {'input': x})[0]

            labels = np.argmax(outputs, axis=-1)
            reference_labels = np.argmax(reference_outputs, axis=-1)
            accuracy = np.sum(labels == reference_labels) / reference_labels.size

            print("Accuracy with respect to reference model: {:.2f}%".format(100. * accuracy))
        else:
            np.savetxt('input.dat', x)
            np.savetxt('output.dat', outputs)

    else:
        assert args.dataset_path
        model_sess = ort.InferenceSession(args.input)
        ref_sess = ort.InferenceSession(args.reference)
        correct = 0
        predictions = 0
        for root, dirs, files in os.walk(args.dataset_path):
            for name in files:
                filename = os.path.join(root, name)
                x = cv2.imread(filename)
                if x is None:
                    continue

                min_shape = min(x.shape[:-1])
                non_rand_resize_scale = 256.0 / min_shape
                new_shape = [round(sh * non_rand_resize_scale) for sh in x.shape[:-1]]
                x = cv2.resize(x, new_shape, interpolation=cv2.INTER_AREA)
                x = x[:, :, [2, 1, 0]]

                indx_min = [int(sh / 2 - 112) for sh in x.shape[:-1]]

                x = x[indx_min[0]:(indx_min[0] + 224), indx_min[1]:(indx_min[1] + 224), :]
                x = (x / 255. - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                x = x.astype(np.float32)
                x = np.transpose(x, (2, 0, 1))
                x = np.expand_dims(x, axis=0)

                outputs = model_sess.run(None, {'input': x})[0]
                reference_outputs = ref_sess.run(None, {'input': x})[0]

                labels = np.argmax(outputs, axis=-1)
                reference_labels = np.argmax(reference_outputs, axis=-1)
                correct += np.sum(labels == reference_labels)
                predictions += x.shape[0]
                accuracy = correct / predictions
                # print(name, labels, reference_labels)
                # print(name, outputs)
                print("Accuracy with respect to reference model: {:.2f}%, {:d}/50,000".format(100. * accuracy, predictions), end='\r')


if __name__ == "__main__":
    main()
