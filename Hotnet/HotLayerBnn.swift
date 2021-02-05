// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate

class HotLayerBnn {
    let cNeuronsThisLayer: Int
    let cNeuronsNextLayerDown: Int

    let activation: BNNSActivation
    let filter: BNNSFilter

    var outputBuffer: UnsafeMutableRawPointer

    init(
        isPoolingLayer: Bool, cNeuronsThisLayer: Int, cNeuronsNextLayerDown: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?,
        outputBuffer: UnsafeMutableRawPointer,
        activation: BNNSActivation = .init(function: .tanh)
    ) {
        self.cNeuronsThisLayer = cNeuronsThisLayer
        self.cNeuronsNextLayerDown = cNeuronsNextLayerDown
        self.outputBuffer = outputBuffer
        self.activation = activation

        let side = Int(sqrt((Double(cNeuronsThisLayer))))

        self.filter = isPoolingLayer ?
            HotLayerBnn.configureInputPoolingLayer(
                kernelWidth: side, kernelHeight: side,
                biases: biases, weights: weights
            )
            
            :

            HotLayerBnn.configureFullyConnectedLayer(
                cNeuronsIn: cNeuronsThisLayer, cNeuronsOut: cNeuronsNextLayerDown,
                biases: biases, weights: weights, activation: activation
            )
    }

    deinit { BNNSFilterDestroy(filter) }

    func activate(inputBuffer: UnsafeRawPointer) {
        BNNSFilterApply(filter, inputBuffer, outputBuffer)
    }
}

extension HotLayerBnn {
    static func getActivation(
        _ standardized: HotNetConfig.Activation
    ) -> BNNSActivation {
        switch standardized {
        case .identity: return BNNSActivation(function: .identity)
        case .tanh:     return BNNSActivation(function: .tanh)
        }
    }
}
