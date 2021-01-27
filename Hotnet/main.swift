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

func blas() {

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

    let blas = HotNetBlas(
        configuration, biases: pBiases, weights: pWeights,
        callbackDispatch: callbackDispatch
    )

    let pInputs_ = UnsafeMutableRawPointer.allocate(
        byteCount: 4 * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    let inputs = [Float]([1, 2, 3, 4])

    pInputs_.initializeMemory(
        as: Float.self, from: inputs, count: 4
    )

    let pInputs = UnsafeRawPointer(pInputs_)

    blas.activate(input: pInputs) {
        print("blasResult \($0.map { $0 })")
        pInputs.deallocate()
        semaphore.signal()
    }
}

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

    let pInputs_ = UnsafeMutableRawPointer.allocate(
        byteCount: 4 * MemoryLayout<Float>.size,
        alignment: MemoryLayout<Float>.alignment
    )

    let inputs = [Float]([1, 2, 3, 4])

    pInputs_.initializeMemory(
        as: Float.self, from: inputs, count: 4
    )

    let pInputs = UnsafeRawPointer(pInputs_)

    bnn.activate(input: pInputs) {
        print("bnnResult \($0.map { $0 })")
        pInputs.deallocate()
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

    cnn.activate(input: [Float]([1, 2, 3, 4])) {
        print("cnnResult \($0.map { $0 })")
        semaphore.signal()
    }
}

blas()
semaphore.wait()

bnn()
semaphore.wait()

cnn()
semaphore.wait()
