// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

struct CnnLayerOutputDescriptor {
    let activationFunction: MPSCNNNeuron?
    let cNeurons: Int

    init(
        _ activationFunction: MPSCNNNeuron? = MPSCNNNeuronTanH(), _ cNeurons: Int
    ) {
        self.activationFunction = activationFunction
        self.cNeurons = cNeurons
    }
}

class HotLayerCnn {
    let cNeuronsIn: Int
    let cNeuronsOut: Int

    let commandQueue: MTLCommandQueue
    weak var device: MTLDevice!
    let isAsync: Bool

    let filter: MPSCNNFullyConnected
    let inputImage: MPSImage
    let outputImage: MPSImage

    var outputBuffer: UnsafeMutableRawPointer

    init(
        device: MTLDevice, isAsync: Bool,
        cNeuronsIn: Int, cNeuronsOut: Int,
        weights: [Float], biases: [Float],
        outputBuffer: UnsafeMutableRawPointer,
        activationFunction: MPSCNNNeuron? = MPSCNNNeuronTanH()
    ) {
        self.cNeuronsIn = cNeuronsIn
        self.cNeuronsOut = cNeuronsOut
        self.device = device
        self.outputBuffer = outputBuffer
        self.isAsync = isAsync

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

    func activate(
        inputBuffer: UnsafeRawPointer,
        _ onComplete: (() -> Void)? = nil
    ) {
        let ib = inputBuffer.bindMemory(to: Float.self, capacity: cNeuronsIn)
        let ic = UnsafeBufferPointer(start: ib, count: cNeuronsIn)
        let input16 = Float16.floats_to_float16s(values: ic.map { $0 })
        
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

        let commandBuffer = commandQueue.makeCommandBuffer()!

        filter.encode(
            commandBuffer: commandBuffer, sourceImage: inputImage,
            destinationImage: outputImage
        )

        commandBuffer.commit()

        if isAsync {
            commandBuffer.addCompletedHandler { _ in
                self.getActivationResult()
                onComplete!()
            }
        } else {
            commandBuffer.waitUntilCompleted()
            getActivationResult()
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
