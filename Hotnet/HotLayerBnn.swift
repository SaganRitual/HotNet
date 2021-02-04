// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate

class HotLayerBnn {
    let cNeuronsIn: Int
    let cNeuronsOut: Int

    let activation: BNNSActivation
    let filter: BNNSFilter

    var outputBuffer: UnsafeMutableRawPointer

    init(
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?,
        outputBuffer: UnsafeMutableRawPointer,
        activation: BNNSActivation = .init(function: .tanh)
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.outputBuffer = outputBuffer
        self.activation = activation

        self.filter = HotLayerBnn.configureFullyConnectedLayer(
            cNeuronsIn: cNeuronsIn, cNeuronsOut: cNeuronsOut,
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
