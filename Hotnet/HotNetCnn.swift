// We are a way for the cosmos to know itself. -- C. Sagan

import Foundation
import MetalPerformanceShaders

class HotNetCnn: HotNet {
    static let theDevice = MTLCopyAllDevices()[0]
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
        _ configuration: HotNetConfig,
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
            configuration.counts.perLayerCounts.last!.cNeurons

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

        let outputImage = intermediateImages.last!
        let width = outputImage.width, height = outputImage.height
        let fc = outputImage.featureChannels
        let count = width * height * fc

        let t = finalOutputBuffer.assumingMemoryBound(to: UInt16.self)
        let f = UnsafeMutableBufferPointer(start: t, count: count)

        let output16_ = UnsafeBufferPointer(f)
        let output16 = output16_.map { $0 }
        let output32 = Float16.float16s_to_floats(values: output16)

        output32.withUnsafeBufferPointer { p in
            super.callbackDispatch.async { [self] in onComplete!(p) }
        }
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
//        let count =  width * height * fc
        let numSlices = (fc + 3)/4

        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: width, height: height, depth: 1))

//        let t = finalOutputBuffer.assumingMemoryBound(to: Float.self)
//        let f = UnsafeMutableBufferPointer(start: t, count: count)

        for i in 0..<numSlices {
          texture.getBytes(finalOutputBuffer,
                           bytesPerRow: width * MemoryLayout<Float>.size,
                           bytesPerImage: 0,
                           from: region,
                           mipmapLevel: 0,
                           slice: i)
        }
    }
}

private extension HotNetCnn {

    static func makeLayers(
        _ configuration: HotNetConfig,
        device: MTLDevice,
        biases: UnsafeMutableRawPointer?,
        weights: UnsafeMutableRawPointer?
    ) -> ([MPSImage], [HotLayerCnn]) {
        var intermediateImages = [MPSImage]()

        let buffers = HotNetConfig.Buffers(configuration, biases, weights)

        let layers: [HotLayerCnn] = (0..<configuration.layerDescriptors.count).map { layerIndex in
            let descriptor = MPSImageDescriptor(
                channelFormat: buffers.isOutputLayer ? .float32 : .float16,
                width: 1, height: 1, featureChannels: buffers.next!.cNeurons
            )

            let image = MPSImage(device: device, imageDescriptor: descriptor)

            intermediateImages.append(image)

            defer { buffers.advanceLayer() }

            let activation = HotLayerCnn.getActivation(configuration.activation)

            return HotLayerCnn(
                device: device,
                cNeuronsIn: buffers.curr.cNeurons,
                cNeuronsOut: buffers.next!.cNeurons,
                biases: buffers.biases(), weights: buffers.weights(),
                outputImage: image, activation: activation
            )
        }

        return (intermediateImages, layers)
    }

}
