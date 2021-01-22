// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

struct CNNLayerFactory: HotLayerFactoryProtocol {
    let device: MTLDevice

    func makeLayer(
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer,
        outputBuffer: UnsafeMutableRawPointer,
        activation: HotNetConfiguration.Activation
    ) -> HotLayerProtocol {
        HotLayerCnn(
            device: self.device, isAsync: false,
            cNeuronsIn: cNeuronsIn, cNeuronsOut: cNeuronsOut,
            biases: biases, weights: weights,
            outputBuffer: outputBuffer,
            activationFunction:
                CNNLayerFactory.getActivation(device, activation)
        )
    }

    static func getActivation(
        _ device: MTLDevice,
        _ standardized: HotNetConfiguration.Activation
    ) -> MPSCNNNeuron? {
        switch standardized {
        case .identity:
            return nil

        case .tanh:
            return MPSCNNNeuron(
                device: device,
                neuronDescriptor: MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .tanH)
            )
        }
    }
}
