// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation
import MetalPerformanceShaders

print("Hello, World!")

let semaphore = DispatchSemaphore(value: 0)

struct LayerOutputDescriptor {
    let activationFunction: BNNSActivation
    let cNeurons: Int

    init(
        _ activationFunction: BNNSActivationFunction = .tanh, _ cNeurons: Int
    ) {
        self.activationFunction = BNNSActivation(function: activationFunction)
        self.cNeurons = cNeurons
    }
}

func bnn() {
    let parameters = (0..<2).flatMap { _ in
        [Float](repeating: 1, count: 16) + [Float](repeating: 0, count: 4)
    }

    let bnn = HotNetBnn(
        layout: [
            LayerOutputDescriptor(.identity, 4),
            LayerOutputDescriptor(.identity, 4),
            LayerOutputDescriptor(.identity, 4)
        ], parameters: parameters
    )

    let bnnResult = bnn.activate(input: [1, 2, 3, 4])
    print("bnnResult \(bnnResult.map { $0 })")

    let pOutput = UnsafeMutableRawPointer.allocate(
        byteCount: 4 * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    let layer = HotLayerBnn(
        cNeuronsIn: 4, cNeuronsOut: 4, weights: [Float](repeating: 1, count: 16), biases: [Float](repeating: 0, count: 4),
        outputBuffer: pOutput, activationFunction: .init(function: .identity)
    )

    layer.activate(inputBuffer: [Float]([1, 2, 3, 4]))

    let t = pOutput.assumingMemoryBound(to: Float.self)
    let f = UnsafeBufferPointer(start: t, count: 4)

    print("layer result \(f.map { $0 })")
}

func cnn() {
    let gpus = MTLCopyAllDevices()
    let device = gpus[1]

    let cNeuronsIn = 4, cNeuronsOut = 4

    let weights = [Float](repeating: 1, count: cNeuronsIn * cNeuronsOut)
    let biases = [Float](repeating: 0, count: cNeuronsOut)

    let descriptor = MPSCNNConvolutionDescriptor(
        kernelWidth: 1, kernelHeight: 1,
        inputFeatureChannels: cNeuronsIn,
        outputFeatureChannels: cNeuronsOut,
        neuronFilter: nil
    )

    let dataSource = DataSourceCnn(
        weights: weights, biases: biases,
        convolutionDescriptor: descriptor
    )

    let filter = MPSCNNFullyConnected(device: device, weights: dataSource)

    // The fully-connected layer does not seem to like .float32 as input,
    // so we'll use .float16. This does mean we have to convert our data
    // from 32-bits to 16-bit floats. For output, it seems float32 is OK.
    let inputImgDesc = MPSImageDescriptor(
        channelFormat: .float16, width: 1, height: 1, featureChannels: cNeuronsIn
    )

    let outputImgDesc = MPSImageDescriptor(
        channelFormat: .float16, width: 1, height: 1, featureChannels: cNeuronsOut
    )

    let inputImage = MPSImage(device: device, imageDescriptor: inputImgDesc)
    let outputImage = MPSImage(device: device, imageDescriptor: outputImgDesc)

    print("inputImage.texture.arrayLength, \(inputImage.texture.arrayLength)")

    let input16 = Bloat16.floats_to_float16s(values: [Float]([1,1,1,1]))
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
    commandBuffer.waitUntilCompleted()

    let width = outputImage.width, height = outputImage.height
    let fc = outputImage.featureChannels, texture = outputImage.texture
    let count =  width * height * fc
    let numSlices = (fc + 3)/4
    var output = [UInt16](repeating: 0, count: count)

    print("output stuff")
    print("w = \(width), h = \(height), fc = \(fc), count = \(count), ns = \(numSlices)")

    let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                           size: MTLSize(width: width, height: height, depth: 1))

    for i in 0..<numSlices {
      texture.getBytes(&(output[width * height * 4 * i]),
                       bytesPerRow: width * 4 * MemoryLayout<UInt16>.size,
                       bytesPerImage: 0,
                       from: region,
                       mipmapLevel: 0,
                       slice: i)
    }

    let final = Bloat16.float16s_to_floats(values: output)

    print("cnn layer result \(final)")

    let ininin = [Float]([1, 2, 3, 4])
    print("ininin \(ininin)")

    let midmidmid = Bloat16.floats_to_float16s(values: ininin)
    print("midmidmid \(midmidmid)")

    let outoutout = Bloat16.float16s_to_floats(values: midmidmid)
    print("outoutout \(outoutout)")

    semaphore.signal()
}

bnn()
cnn()
semaphore.wait()
