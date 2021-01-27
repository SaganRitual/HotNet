// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class HotLayerCnn {
    let cNeuronsIn: Int
    let cNeuronsOut: Int

    let filter: MPSCNNFullyConnected
    weak var outputImage: MPSImage!

    weak var device: MTLDevice!

    init(
        device: MTLDevice,
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?, weights: UnsafeMutableRawPointer?,
        outputImage: MPSImage,
        activation: MPSCNNNeuron? = MPSCNNNeuronTanH()
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.device = device
        self.outputImage = outputImage

        let descriptor = MPSCNNConvolutionDescriptor(
            kernelWidth: 1, kernelHeight: 1,
            inputFeatureChannels: cNeuronsIn,
            outputFeatureChannels: cNeuronsOut,
            neuronFilter: activation
        )

        let dataSource = DataSourceCnn(
            biases: biases, weights: weights,
            convolutionDescriptor: descriptor
        )

        self.filter = MPSCNNFullyConnected(device: device, weights: dataSource)
    }

    func encodeActivation(inputImage: MPSImage, commandBuffer: MTLCommandBuffer) {
        filter.encode(
            commandBuffer: commandBuffer, sourceImage: inputImage,
            destinationImage: outputImage
        )
    }
}

extension HotLayerCnn {
    static func getActivation(
        _ standardized: HotNetConfiguration.Activation
    ) -> MPSCNNNeuron? {
        switch standardized {
        case .identity:
            return nil

        case .tanh:
            return MPSCNNNeuron(
                device: HotNetCnn.theDevice,
                neuronDescriptor: MPSNNNeuronDescriptor.cnnNeuronDescriptor(with: .tanH)
            )
        }
    }
}
