// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation
import MetalPerformanceShaders

print("Hello, World!")

let semaphore = DispatchSemaphore(value: 0)

func bnn() {

    let cWeights = 32
    let cBiases = 8

    let pWeights = UnsafeMutableRawPointer.allocate(
        byteCount: cWeights * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    pWeights.initializeMemory(as: Float.self, repeating: 1, count: cWeights)

    let pBiases = UnsafeMutableRawPointer.allocate(
        byteCount: cBiases * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    pBiases.initializeMemory(as: Float.self, repeating: 0, count: cBiases)

    let configuration = HotNetConfiguration(
        activation: .identity, isAsync: true,
        layerDescriptors: [
            .init(cNeurons: 4), .init(cNeurons: 4), .init(cNeurons: 4)
        ],
        layerFactory: BNNLayerFactory()
    )

    let bnn = HotNet(configuration, biases: pBiases, weights: pWeights)

    bnn.activate(input: [1, 2, 3, 4]) {
        print("bnnResult \($0.map { $0 })")
    }

//    let pOutput = UnsafeMutableRawPointer.allocate(
//        byteCount: 4 * MemoryLayout<Float>.size,
//        alignment: MemoryLayout<Float>.alignment
//    )
//
//    let layer = HotLayerBnn(
//        cNeuronsIn: 4, cNeuronsOut: 4,
//        weights: [Float](repeating: 1, count: 16),
//        biases: [Float](repeating: 0, count: 4),
//        outputBuffer: pOutput,
//        activationFunction: .init(function: .identity)
//    )
//
//    layer.activate(inputBuffer: [Float]([1, 2, 3, 4]))
//
//    let t = pOutput.assumingMemoryBound(to: Float.self)
//    let f = UnsafeBufferPointer(start: t, count: 4)
//
//    print("bnn layer result \(f.map { $0 })")
}

func cnn() {
    let gpus = MTLCopyAllDevices()
    let device = gpus[1]

    let cWeights = 32
    let cBiases = 8

    let pWeights = UnsafeMutableRawPointer.allocate(
        byteCount: cWeights * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    pWeights.initializeMemory(as: Float.self, repeating: 1, count: cWeights)

    let pBiases = UnsafeMutableRawPointer.allocate(
        byteCount: cBiases * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    pBiases.initializeMemory(as: Float.self, repeating: 0, count: cBiases)

    let configuration = HotNetConfiguration(
        activation: .identity, isAsync: true,
        layerDescriptors: [
            .init(cNeurons: 4), .init(cNeurons: 4), .init(cNeurons: 4)
        ],
        layerFactory: CNNLayerFactory(device: device)
    )

    let cnn = HotNet(configuration, biases: pBiases, weights: pWeights)

    cnn.activate(input: [1, 2, 3, 4]) {
        print("cnnResult \($0.map { $0 })")
        semaphore.signal()
    }

//
//    let pOutput = UnsafeMutableRawPointer.allocate(
//        byteCount: 4 * MemoryLayout<Float>.size,
//        alignment: MemoryLayout<Float>.alignment
//    )
//
//    let layer = HotLayerCnn(
//        device: device, isAsync: false,
//        cNeuronsIn: 4, cNeuronsOut: 4,
//        weights: [Float](repeating: 1, count: 16),
//        biases: [Float](repeating: 0, count: 4),
//        outputBuffer: pOutput, activationFunction: nil
//    )
//
//    layer.activate(inputBuffer: [Float]([1, 2, 3, 4]))
//
//    let t = pOutput.assumingMemoryBound(to: Float.self)
//    let f = UnsafeBufferPointer(start: t, count: 4)
//
//    print("cnn layer result \(f.map { $0 })")
}

bnn()
cnn()
semaphore.wait()
