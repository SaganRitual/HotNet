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

let pInputs_ = UnsafeMutableRawPointer.allocate(
    byteCount: 240 * MemoryLayout<Float>.size,
    alignment: MemoryLayout<Float>.alignment
)

let inputs = [Float](repeating: 1, count: 240)

pInputs_.initializeMemory(
    as: Float.self, from: inputs, count: 240
)

let pInputs = UnsafeRawPointer(pInputs_)

let cWeights = 240 * 180 + 180 * 120
let cBiases = 240 + 180

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
        .init(cNeurons: 240), .init(cNeurons: 180), .init(cNeurons: 120)
    ]
)

func blas() {

    let blas = HotNetBlas(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    blas.activate(input: pInputs) {
        print("blasResult \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

func bnn() {

    let bnn = HotNetBnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    bnn.activate(input: pInputs) {
        print("bnnResult \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

func cnn() {

    let cnn = HotNetCnn(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    cnn.activate(input: [Float]([1, 2, 3, 4])) {
        print("cnnResult \($0.reduce(0) { $0 + $1 })")
        semaphore.signal()
    }
}

blas()
semaphore.wait()

bnn()
semaphore.wait()

cnn()
semaphore.wait()
