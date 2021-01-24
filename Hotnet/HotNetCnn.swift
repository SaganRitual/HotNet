// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class HotNetCnn: HotNet {
    static let theDevice = MTLCopyAllDevices()[1]
    static let theCommandQueue = theDevice.makeCommandQueue()!

    let cNeuronsIn: Int
    let float16Transfer: UnsafeMutableRawPointer
    let float16Buffer: UnsafeMutableBufferPointer<UInt16>

    let intermediateImages: [MPSImage]
    let layers: [HotLayerCnn]

    let origin: MTLOrigin
    let region: MTLRegion
    let size: MTLSize

    let inputImageDescriptor: MPSImageDescriptor
    let inputImage: MPSImage

    let finalOutputBuffer: UnsafeMutableRawPointer

    deinit {
        finalOutputBuffer.deallocate()
        float16Transfer.deallocate()
    }

    init(
        _ configuration: HotNetConfiguration,
        biases: UnsafeMutableRawPointer,
        weights: UnsafeMutableRawPointer,
        callbackDispatch: DispatchQueue = DispatchQueue.main
    ) {
        (intermediateImages, layers) = HotNetCnn.makeLayers(
            configuration,
            device: HotNetCnn.theDevice,
            biases: biases, weights: weights
        )

        let finalOutputNeurons =
            configuration.layerDescriptors.last!.cNeurons

        self.finalOutputBuffer = UnsafeMutableRawPointer.allocate(
            byteCount: finalOutputNeurons * MemoryLayout<Float>.size,
            alignment: MemoryLayout<Float>.alignment
        )

        // Point directly to the output buffer of the bottom layer, so
        // caller can read it without any copying
        let t = finalOutputBuffer.bindMemory(
            to: Float.self, capacity: finalOutputNeurons
        )

        let outputBuffer = UnsafeBufferPointer(
            start: t, count: configuration.layerDescriptors.last!.cNeurons
        )

        self.cNeuronsIn = layers.first!.cNeuronsIn

        self.float16Transfer = UnsafeMutableRawPointer.allocate(
            byteCount: self.cNeuronsIn * MemoryLayout<UInt16>.size,
            alignment: MemoryLayout<UInt16>.alignment
        )

        let u = float16Transfer.bindMemory(to: UInt16.self, capacity: self.cNeuronsIn)
        self.float16Buffer = UnsafeMutableBufferPointer(start: u, count: self.cNeuronsIn)

        self.inputImageDescriptor = MPSImageDescriptor(
            channelFormat: .float16, width: 1, height: 1,
            featureChannels: cNeuronsIn
        )

        self.inputImage = MPSImage(
            device: HotNetCnn.theDevice, imageDescriptor: inputImageDescriptor
        )

        self.origin = MTLOriginMake(0, 0, 0)
        self.size = MTLSizeMake(1, 1, 1)
        self.region = MTLRegion(origin: origin, size: size)

        super.init(outputBuffer: outputBuffer, callbackDispatch: callbackDispatch)
    }

    var onComplete: ((UnsafeBufferPointer<Float>) -> Void)?

    override func activate(
        input: UnsafeRawPointer,
        _ onComplete: @escaping (UnsafeBufferPointer<Float>) -> Void
    ) {
        let commandBuffer = HotNetCnn.theCommandQueue.makeCommandBuffer()!

        commandBuffer.addCompletedHandler(onActivationComplete)

        self.onComplete = onComplete

        let inputImage = setupInputBuffer(input)

        layers.indices.forEach {
            let layer = layers[$0]

            if $0 == 0 {
                layer.encodeActivation(
                    inputImage: inputImage, commandBuffer: commandBuffer
                )
            } else {
                layer.encodeActivation(
                    inputImage: intermediateImages[$0 - 1],
                    commandBuffer: commandBuffer
                )
            }
        }

        commandBuffer.commit()
    }

    func onActivationComplete(_: MTLCommandBuffer) {
        getActivationResult()
        super.callbackDispatch.async { [self] in onComplete!(outputBuffer) }
    }

    func setupInputBuffer(_ floatBuffer: UnsafeRawPointer) -> MPSImage {
        Float16.floats_to_float16s(input: floatBuffer, output: float16Buffer)

        inputImage.texture.replace(
            region: region, mipmapLevel: 0,
            withBytes: float16Transfer,
            bytesPerRow: MemoryLayout<UInt16>.stride * 4
        )

        return inputImage
    }

    func getActivationResult() {
        let outputImage = intermediateImages.last!
        let width = outputImage.width, height = outputImage.height
        let fc = outputImage.featureChannels, texture = outputImage.texture
        let count =  width * height * fc
        let numSlices = (fc + 3)/4

        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: width, height: height, depth: 1))

        let t = finalOutputBuffer.assumingMemoryBound(to: Float.self)
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

private extension HotNetCnn {
    static func isOutputLayer(
        _ index: Int, _ configuration: HotNetConfiguration
    ) -> Bool {
        index >= configuration.layerDescriptors.count - 1
    }

    static func makeLayers(
        _ configuration: HotNetConfiguration,
        device: MTLDevice,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?
    ) -> ([MPSImage], [HotLayerCnn]) {
        var intermediateImages = [MPSImage]()
        var pBiases = biases
        var pWeights = weights

        let layers: [HotLayerCnn] = zip(
            configuration.layerDescriptors.enumerated().dropLast(),
            configuration.layerDescriptors.enumerated().dropFirst()
        ).map {
            let (upperLayerIndex, inputs) = $0
            let (lowerLayerIndex, outputs) = $1

            let cWeights = inputs.cNeurons * outputs.cNeurons
            let cBiases = outputs.cNeurons

            let isOutputLayer = HotNetCnn.isOutputLayer(
                lowerLayerIndex, configuration
            )

            let descriptor = MPSImageDescriptor(
                channelFormat: isOutputLayer ? .float32 : .float16,
                width: 1, height: 1, featureChannels: outputs.cNeurons
            )

            let image = MPSImage(device: device, imageDescriptor: descriptor)

            intermediateImages.append(image)

            defer {
                if pWeights != nil && upperLayerIndex == 0 {
                    pWeights! += cWeights * MemoryLayout<Float>.size
                }

                if pBiases != nil &&
                    !HotNetCnn.isOutputLayer(lowerLayerIndex, configuration) {
                    pBiases! += cBiases * MemoryLayout<Float>.size
                }
            }

            let activation = CNNLayerFactory.getActivation(
                configuration.activation
            )

            return HotLayerCnn(
                device: device,
                cNeuronsIn: inputs.cNeurons, cNeuronsOut: outputs.cNeurons,
                biases: biases, weights: weights,
                outputImage: image,
                activation: activation

            )
        }

        return (intermediateImages, layers)
    }

}
