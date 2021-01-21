// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation

struct BNNLayerFactory: HotLayerFactoryProtocol {
    func makeLayer(
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer,
        outputBuffer: UnsafeMutableRawPointer,
        activation: HotNetConfiguration.Activation
    ) -> HotLayerProtocol {
        HotLayerBnn(
            cNeuronsIn: cNeuronsIn, cNeuronsOut: cNeuronsOut,
            biases: biases, weights: weights,
            outputBuffer: outputBuffer,
            activationFunction: BNNLayerFactory.getActivation(activation)
        )

    }

    static func getActivation(
        _ standardized: HotNetConfiguration.Activation
    ) -> BNNSActivation {
        switch standardized {
        case .identity: return BNNSActivation(function: .identity)
        case .tanh:     return BNNSActivation(function: .tanh)
        }
    }
}
