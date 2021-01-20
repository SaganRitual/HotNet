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

    weak var device: MTLDevice!
    let commandQueue: MTLCommandQueue

    let filter: MPSCNNFullyConnected
    let inputImage: MPSImage
    let outputImage: MPSImage

    var outputBuffer: UnsafeMutableRawPointer

    init(
        device: MTLDevice,
        cNeuronsIn: Int, cNeuronsOut: Int,
        weights: [Float], biases: [Float],
        outputBuffer: UnsafeMutableRawPointer,
        activationFunction: MPSCNNNeuron? = MPSCNNNeuronTanH()
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.device = device
        self.outputBuffer = outputBuffer

        self.commandQueue = device.makeCommandQueue()!

        let descriptor = MPSCNNConvolutionDescriptor(
            kernelWidth: 1, kernelHeight: 1,
            inputFeatureChannels: cNeuronsIn,
            outputFeatureChannels: cNeuronsOut,
            neuronFilter: activationFunction
        )

        let dataSource = DataSourceCnn(
            weights: weights, biases: biases,
            convolutionDescriptor: descriptor
        )

        self.filter = MPSCNNFullyConnected(device: device, weights: dataSource)

        // The fully-connected layer does not seem to like .float32 as input,
        // so we'll use .float16. This does mean we have to convert our data
        // from 32-bits to 16-bit floats. For output, it seems float32 is OK.
        let inputImgDesc = MPSImageDescriptor(
            channelFormat: .float16, width: 1, height: 1, featureChannels: cNeuronsIn
        )

        let outputImgDesc = MPSImageDescriptor(
            channelFormat: .float32, width: 1, height: 1, featureChannels: cNeuronsOut
        )

        self.inputImage = MPSImage(device: device, imageDescriptor: inputImgDesc)
        self.outputImage = MPSImage(device: device, imageDescriptor: outputImgDesc)
    }

    func activate(inputBuffer: [Float]) {
        activate(inputBuffer: inputBuffer, waitForResult: true)
    }

    func activate(
        inputBuffer: UnsafeRawPointer,
        _ onComplete: (() -> Void)? = nil,
        waitForResult: Bool = false
    ) {
        let ib = inputBuffer.bindMemory(to: Float.self, capacity: cNeuronsIn)
        let ic = UnsafeBufferPointer(start: ib, count: cNeuronsIn)
        let input16 = Bloat16.floats_to_float16s(values: ic.map { $0 })
        
        input16.withUnsafeBufferPointer { ptr in
          for i in 0..<inputImage.texture.arrayLength {
            let region = MTLRegion(origin: MTLOriginMake(0, 0, 0), size: MTLSizeMake(1, 1, 1))
            inputImage.texture.replace(
                region: region, mipmapLevel: 0, slice: i,
                withBytes: ptr.baseAddress!.advanced(by: i * 4),
                bytesPerRow: MemoryLayout<UInt16>.stride * 4, bytesPerImage: 0
            )
          }
        }

        let commandQueue = device.makeCommandQueue()!
        let commandBuffer = commandQueue.makeCommandBuffer()!

        filter.encode(
            commandBuffer: commandBuffer, sourceImage: inputImage,
            destinationImage: outputImage
        )

        commandBuffer.commit()

        if waitForResult {
            commandBuffer.waitUntilCompleted()
            getActivationResult()
        } else {
            commandBuffer.addCompletedHandler { _ in
                self.getActivationResult()
                onComplete!()
            }
        }
    }

    private func getActivationResult() {
        let width = outputImage.width, height = outputImage.height
        let fc = outputImage.featureChannels, texture = outputImage.texture
        let count =  width * height * fc
        let numSlices = (fc + 3)/4

        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: width, height: height, depth: 1))

        let t = outputBuffer.assumingMemoryBound(to: Float.self)
        let f = UnsafeMutableBufferPointer(start: t, count: count)

        for i in 0..<numSlices {
          texture.getBytes(&(f[width * height * 4 * i]),
                           bytesPerRow: width * 4 * MemoryLayout<Float>.size,
                           bytesPerImage: 0,
                           from: region,
                           mipmapLevel: 0,
                           slice: i)
        }
    }
}
