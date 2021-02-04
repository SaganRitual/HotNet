// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation
import MetalPerformanceShaders

print("Hello, World!")

let semaphore = DispatchSemaphore(value: 0)
let theDevice = MTLCopyAllDevices()[0]

let callbackDispatch = DispatchQueue(
    label: "callback", attributes: .concurrent, target: DispatchQueue.global()
)

func fAlign() -> Int { MemoryLayout<Float>.alignment }
func fBytes(cElements: Int = 1) -> Int { MemoryLayout<Float>.size * cElements }

let configuration = HotNetConfig(
    activation: .identity,
    layerDescriptors: [
        .init(cNeurons: 2), .init(cNeurons: 2), .init(cNeurons: 1)
    ]
)

var cWeights: Int { configuration.counts.cWeights }
var cBiases: Int { configuration.counts.cBiases }

let pInputs_ = UnsafeMutableRawPointer.allocate(
    byteCount: fBytes(cElements: configuration.counts.cNeurons),
    alignment: fAlign()
)

pInputs_.initializeMemory(
    as: Float.self, repeating: 1, count: configuration.counts.cNeurons
)

let pInputs = UnsafeRawPointer(pInputs_)

let pWeights = UnsafeMutableRawPointer.allocate(
    byteCount: fBytes(cElements: cWeights), alignment: fAlign()
)

pWeights.initializeMemory(as: Float.self, repeating: 1, count: cWeights)

let pBiases = UnsafeMutableRawPointer.allocate(
    byteCount: fBytes(cElements: cBiases), alignment: fAlign()
)

pBiases.initializeMemory(as: Float.self, repeating: 0, count: cBiases)

func blas() {

    let blas = HotNetBlas(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    blas.activate(input: pInputs) {
        print("blasResult \($0.map { $0 }) \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

func bnn() {

    let bnn = HotNetBnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    bnn.activate(input: pInputs) {
        print("bnnResult \($0.map { $0 }) \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

func cnn() {

    let cnn = HotNetCnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    cnn.activate(input: pInputs) {
        print("cnnResult \($0.map { $0 }) \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

blas()
semaphore.wait()

bnn()
semaphore.wait()

cnn()
semaphore.wait()

let float16Transfer = UnsafeMutableRawPointer.allocate(
    byteCount: 1 * MemoryLayout<UInt16>.size,
    alignment: MemoryLayout<UInt16>.alignment
)

let v = float16Transfer.bindMemory(to: UInt16.self, capacity: 1)
let float16Buffer = UnsafeMutableBufferPointer(start: v, count: 1)

let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                       size: MTLSize(width: 1, height: 1, depth: 1))

func setupInputBuffer(_ floatBuffer: UnsafeRawPointer) -> MPSImage {
    Float16.floats_to_float16s(input: floatBuffer, output: float16Buffer)

    inputImage_.texture.replace(
        region: region, mipmapLevel: 0,
        withBytes: float16Transfer,
        bytesPerRow: MemoryLayout<UInt16>.stride * 4
    )

    return inputImage_
}

let descriptor = MPSImageDescriptor(
    channelFormat: .float16,
    width: 1, height: 1, featureChannels: 1
)

let floatBuffer = UnsafeMutableRawPointer.allocate(
    byteCount: 1 * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment
)

floatBuffer.initializeMemory(as: Float.self, repeating: 1, count: 1)

let inputImage_ = MPSImage(device: theDevice, imageDescriptor: descriptor)
let inputImage = setupInputBuffer(floatBuffer)

let outputImage = MPSImage(device: theDevice, imageDescriptor: descriptor)

var weights32 = UnsafeMutableRawPointer.allocate(byteCount: 1 * MemoryLayout<Float>.size, alignment: MemoryLayout<Float>.alignment)
weights32.initializeMemory(as: Float.self, repeating: 1, count: 1)

let weights16Transfer = UnsafeMutableRawPointer.allocate(
    byteCount: 1 * MemoryLayout<UInt16>.size,
    alignment: MemoryLayout<UInt16>.alignment
)

let www = weights16Transfer.bindMemory(to: UInt16.self, capacity: 1)
let weights16Buffer = UnsafeMutableBufferPointer(start: www, count: 1)

Float16.floats_to_float16s(input: weights32, output: weights16Buffer)

let theLayer = HotLayerCnn(
    device: theDevice, cNeuronsIn: 1, cNeuronsOut: 1,
    biases: nil, weights: weights32, outputImage: outputImage,
    activation: nil
)

let cq = theDevice.makeCommandQueue()!
let cb = cq.makeCommandBuffer()!

cb.addCompletedHandler { _ in
    semaphore.signal()
}

theLayer.encodeActivation(inputImage: inputImage, commandBuffer: cb)

cb.commit()
cb.waitUntilCompleted()

let width = 1, height = 1
let fc = 1, texture = outputImage.texture
let count =  width * height * fc
let numSlices = (fc + 3)/4

let finalOutputBuffer = UnsafeMutableRawPointer.allocate(
    byteCount: 1 * MemoryLayout<UInt16>.size,
    alignment: MemoryLayout<UInt16>.alignment
)

// Point directly to the output buffer of the bottom layer, so
// caller can read it without any copying

let t = finalOutputBuffer.assumingMemoryBound(to: UInt16.self)
let f = UnsafeMutableBufferPointer(start: t, count: count)

for i in 0..<numSlices {
  texture.getBytes(finalOutputBuffer,
                   bytesPerRow: width * MemoryLayout<UInt16>.size,
                   bytesPerImage: 0,
                   from: region,
                   mipmapLevel: 0,
                   slice: i)
}

let output16_ = UnsafeBufferPointer(f)
let output16 = output16_.map { $0 }
let output32 = Float16.float16s_to_floats(values: output16)

semaphore.wait()
print("cnn2Result \(output32.map { $0 }) \(output32.reduce(0) { $0 + $1 })")
