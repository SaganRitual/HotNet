// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class DataSourceCnn: NSObject, MPSCNNConvolutionDataSource {
    let pBiases: UnsafeMutablePointer<Float>
    let pWeights: UnsafeMutableRawPointer

    let convolutionDescriptor: MPSCNNConvolutionDescriptor

    init(
        weights: [Float], biases: [Float],
        convolutionDescriptor: MPSCNNConvolutionDescriptor
    ) {
        self.pWeights = UnsafeMutableRawPointer.allocate(
            byteCount: weights.count * MemoryLayout<Float>.size,
            alignment: MemoryLayout<Float>.alignment
        )

        self.pWeights.initializeMemory(
            as: Float.self, from: weights, count: weights.count
        )

        self.pBiases = UnsafeMutablePointer.allocate(capacity: biases.count)
        self.pBiases.initialize(from: biases, count: biases.count)

        self.convolutionDescriptor = convolutionDescriptor
    }

    func dataType() ->   MPSDataType { .float32 }
    func descriptor() -> MPSCNNConvolutionDescriptor { convolutionDescriptor }
    func weights() ->    UnsafeMutableRawPointer { pWeights }
    func biasTerms() ->  UnsafeMutablePointer<Float>? { pBiases }

    func load() -> Bool { true }

    func purge() { }

    func label() -> String? { nil }

    func copy(with zone: NSZone? = nil) -> Any { false }

}

class HotLayerCnn {
    let cNeuronsIn: Int
    let cNeuronsOut: Int

    let dataSource: DataSourceCnn

    weak var device: MTLDevice!
    let commandQueue: MTLCommandQueue

    let filter: MPSCNNFullyConnected
    weak var outputImage: MPSImage!

    init(
        cNeuronsIn: Int, cNeuronsOut: Int,
        weights: [Float], biases: [Float],
        device: MTLDevice, outputImage: MPSImage
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.outputImage = outputImage
        self.device = device

        self.commandQueue = device.makeCommandQueue()!

        let convolutionDescriptor = MPSCNNConvolutionDescriptor(
            kernelWidth: 1, kernelHeight: 1,
            inputFeatureChannels: cNeuronsIn, outputFeatureChannels: cNeuronsOut,
            neuronFilter: nil
        )

        self.dataSource = DataSourceCnn(
            weights: weights, biases: biases,
            convolutionDescriptor: convolutionDescriptor
        )

        self.filter = MPSCNNFullyConnected(device: device, weights: dataSource)
    }

    func activate(
        inputBuffer: UnsafeRawPointer,
        _ onComplete: @escaping () -> Void
    ) {
        let inputImage = convertInputToTexture(inputBuffer)

        let commandBuffer = commandQueue.makeCommandBuffer()!
//        inputImage.synchronize(on: commandBuffer)

        filter.encode(
            commandBuffer: commandBuffer, sourceImage: inputImage,
            destinationImage: outputImage
        )

        commandBuffer.addCompletedHandler { _ in onComplete() }
        commandBuffer.commit()
    }
}

private extension HotLayerCnn {
    func convertInputToTexture(_ input: UnsafeRawPointer) -> MPSImage {
        let imageDescriptor = MPSImageDescriptor(
            channelFormat: .float16, width: 1, height: 1,
            featureChannels: cNeuronsOut
        )

        let inputImage = MPSImage(
            device: device, imageDescriptor: imageDescriptor
        )

        let t = input.bindMemory(to: Float.self, capacity: cNeuronsIn)
        let f = UnsafeBufferPointer(start: t, count: cNeuronsIn)

        for i in 0..<inputImage.texture.arrayLength {
            let region = MTLRegion(
                origin: MTLOriginMake(0, 0, 0), size: MTLSizeMake(1, 1, 1)
            )

            inputImage.texture.replace(
                region: region, mipmapLevel: 0, slice: i,
                withBytes: f.baseAddress!.advanced(by: i * 4),
                bytesPerRow: MemoryLayout<Float>.stride * 4,
                bytesPerImage: 0
            )
        }

        return inputImage
    }
}
