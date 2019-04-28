import TensorFlow

let layerToTensorMapping = [
    "block1_conv1": "layer_with_weights-0",
    "block1_conv2": "layer_with_weights-1",
    "block2_conv1": "layer_with_weights-2",
    "block2_conv2": "layer_with_weights-3",
    "block3_conv1": "layer_with_weights-4",
    "block3_conv2": "layer_with_weights-5",
    "block3_conv3": "layer_with_weights-6",
    "block3_conv4": "layer_with_weights-7",
    "block4_conv1": "layer_with_weights-8",
    "block4_conv2": "layer_with_weights-9",
    "block4_conv3": "layer_with_weights-10",
    "block4_conv4": "layer_with_weights-11",
    "block5_conv1": "layer_with_weights-12",
    "block5_conv2": "layer_with_weights-13",
    "block5_conv3": "layer_with_weights-14",
    "block5_conv4": "layer_with_weights-15"
]

func loadParameters<Scalar: FloatingPoint>(for tensorName: String) -> Tensor<Scalar> {
    let suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    let fullTensorName = StringTensor(["\(tensorName)\(suffix)"])

    return Raw.restoreV2(prefix: StringTensor("./vgg-19.ckpt"), 
              tensorNames: fullTensorName, 
              shapeAndSlices: StringTensor([""]), 
              dtypes: [Float.tensorFlowDataType])[0] as! Tensor<Scalar>
}

public extension Conv2D where Scalar: FloatingPoint {
    init(named name: String) {
        self.init(filter: loadParameters(for: "\(layerToTensorMapping[name]!)/kernel"),
                  bias: loadParameters(for: "\(layerToTensorMapping[name]!)/bias"), 
                  activation: relu,
                  strides: (1, 1),
                  padding: .same)
    }
}
