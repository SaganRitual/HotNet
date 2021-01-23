// We are a way for the cosmos to know itself. -- C. Sagan

import Accelerate
import Foundation
import MetalPerformanceShaders

print("Hello, World!")

let semaphore = DispatchSemaphore(value: 0)
let theDevice = MTLCopyAllDevices()[1]

let callbackDispatch = DispatchQueue(
    label: "callback", attributes: .concurrent, target: DispatchQueue.global()
)

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
        activation: .identity,
        layerDescriptors: [
            .init(cNeurons: 4), .init(cNeurons: 4), .init(cNeurons: 4)
        ]
    )

    let bnn = HotNetBnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    bnn.activate(input: [1, 2, 3, 4]) {
        print("bnnResult \($0.map { $0 })")
        semaphore.signal()
    }
}

func cnn() {
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
        activation: .identity,
        layerDescriptors: [
            .init(cNeurons: 4), .init(cNeurons: 4), .init(cNeurons: 4)
        ]
    )

    let cnn = HotNetCnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    cnn.activate(input: [1, 2, 3, 4]) {
        print("cnnResult \($0.map { $0 })")
        semaphore.signal()
    }
}

bnn()

for _ in 0..<100 { cnn() }

for _ in 0..<100 { semaphore.wait() }
semaphore.wait()
