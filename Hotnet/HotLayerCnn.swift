// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class HotLayerCnn {
    let cNeuronsIn: Int
    let cNeuronsOut: Int

    let filter: MPSCNNFullyConnected
    weak var outputImage: MPSImage!

    weak var device: MTLDevice!
    weak var commandBuffer: MTLCommandBuffer!

    init(
        device: MTLDevice, commandBuffer: MTLCommandBuffer,
        cNeuronsIn: Int, cNeuronsOut: Int,
        biases: UnsafeMutableRawPointer?, weights: UnsafeMutableRawPointer?,
        outputImage: MPSImage,
        activation: MPSCNNNeuron? = MPSCNNNeuronTanH()
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.device = device
        self.commandBuffer = commandBuffer
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

    func encodeActivation(inputImage: MPSImage) {
        filter.encode(
            commandBuffer: commandBuffer, sourceImage: inputImage,
            destinationImage: outputImage
        )
    }
}
